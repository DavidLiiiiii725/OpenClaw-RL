#!/bin/bash
# MathTutor-RL: DeepSeek-Math-7B full fine-tuning with GRPO + SymPy verifier
#
# Key differences from openclaw_rl:
#   - No external PRM server needed: rewards computed via SymPy (built-in)
#   - Rollout function: math_tutor_rollout.generate_rollout_math_tutor
#   - Model: DeepSeek-Math-7B-Instruct (deepseek-ai/deepseek-math-7b-instruct)
#   - Memory files written to MATH_TUTOR_MEMORY_DIR (default: ./math_tutor_memory)
#   - Optional KC-aware OPD: set OPD_ENABLE=1 and provide a separate judge model
#
# GPU layout (8 GPUs default):
#   4 actor GPUs  (Megatron training)
#   2 rollout GPUs (SGLang inference)
#   2 OPD judge GPUs (optional; only used when OPD_ENABLE=1)

SKIP_CLUSTER_CLEANUP=${SKIP_CLUSTER_CLEANUP:-0}
if [ "${SKIP_CLUSTER_CLEANUP}" != "1" ]; then
  pkill -9 sglang
  sleep 3
  ray stop --force
  pkill -9 ray
  pkill -9 python
  sleep 3
  pkill -9 ray
  pkill -9 python
fi

set -ex

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/tmp}"

NUM_GPUS=${NUM_GPUS:-8}
ACTOR_GPUS=${ACTOR_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-2}
# OPD judge GPUs are only allocated when OPD_ENABLE=1
OPD_ENABLE=${OPD_ENABLE:-0}
OPD_GPUS=${OPD_GPUS:-2}

if [ "${OPD_ENABLE}" = "1" ]; then
  REQUIRED_GPUS=$(( ACTOR_GPUS + ROLLOUT_GPUS + OPD_GPUS ))
else
  REQUIRED_GPUS=$(( ACTOR_GPUS + ROLLOUT_GPUS ))
fi

if (( REQUIRED_GPUS > NUM_GPUS )); then
    echo "Required GPUs (${REQUIRED_GPUS}) > available GPUs (${NUM_GPUS})"
    exit 1
fi

export RAY_health_check_failure_threshold=20
export RAY_health_check_period_ms=5000
export RAY_health_check_timeout_ms=30000
export RAY_num_heartbeats_timeout=60

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="$(cd -- "${SCRIPT_DIR}/../slime" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

HF_CKPT=${HF_CKPT:-${REPO_ROOT}/models/deepseek-math-7b-instruct}
REF_LOAD=${REF_LOAD:-${HF_CKPT}}
SAVE_CKPT=${SAVE_CKPT:-${REPO_ROOT}/ckpt/deepseek-math-7b-math-tutor-rl}
OPD_MODEL_PATH=${OPD_MODEL_PATH:-${HF_CKPT}}

# MathTutor-specific environment variables
export SERVED_MODEL_NAME="deepseek-math-7b"
export HOST="0.0.0.0"
export PORT="30000"
export MATH_TUTOR_MEMORY_DIR="${MATH_TUTOR_MEMORY_DIR:-${SCRIPT_DIR}/math_tutor_memory}"
export MATH_TUTOR_RECORD_ENABLED="${MATH_TUTOR_RECORD_ENABLED:-1}"
export MATH_TUTOR_RECORD_FILE="${SCRIPT_DIR}/results/deepseek_math_7b_record.jsonl"
export TP="2"
export CONTEXT_LENGTH="32768"
export MEM_FRACTION_STATIC="0.85"
# DeepSeek-Math uses standard (non-reasoning) chat template
export REASONING_PARSER="${REASONING_PARSER:-}"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-}"

mkdir -p "$(dirname "${MATH_TUTOR_RECORD_FILE}")"
mkdir -p "${MATH_TUTOR_MEMORY_DIR}"

# -----------------------------------------------------------------------
# Model architecture arguments for DeepSeek-Math-7B
# -----------------------------------------------------------------------
MODEL_ARGS=(
  --num-layers 30
  --hidden-size 4096
  --ffn-hidden-size 11008
  --num-attention-heads 32
  --max-position-embeddings 4096
  --norm-epsilon 1e-5
  --normalization RMSNorm
  --position-embedding-type rope
  --rotary-base 10000
  --swiglu
  --untie-embeddings-and-output-weights
  --vocab-size 102400
  --bf16
  --use-flash-attn
)

CKPT_ARGS=(
  --megatron-to-hf-mode bridge
  --hf-checkpoint "${HF_CKPT}"
  --ref-load "${REF_LOAD}"
  --save "${SAVE_CKPT}"
  --save-interval 100
)

# -----------------------------------------------------------------------
# Rollout: request-driven via MathTutorAPIServer
# -----------------------------------------------------------------------
ROLLOUT_ARGS=(
  --disable-rollout-global-dataset
  --rollout-function-path math_tutor_rollout.generate_rollout_math_tutor

  --num-rollout 100000000
  # 64 interactions per GRPO update (as specified in paper)
  --rollout-batch-size 64
  --n-samples-per-prompt 1
  --rollout-max-response-len 8192
  --rollout-max-context-len 32768
  --rollout-temperature 0.6
  --reward-key score

  --num-steps-per-rollout 1
)

# -----------------------------------------------------------------------
# GRPO (Critic-free RL — ~40% GPU memory saving vs PPO)
# -----------------------------------------------------------------------
GRPO_ARGS=(
  --advantage-estimator grpo
  --disable-rewards-normalization
  --use-kl-loss
  --kl-loss-coef 0.01
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
)

# -----------------------------------------------------------------------
# Training performance
# -----------------------------------------------------------------------
PERF_ARGS=(
  --tensor-model-parallel-size 4
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1

  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1

  --use-dynamic-batch-size
  --max-tokens-per-gpu 32768
  --log-probs-chunk-size 1024
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-5
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

EVAL_ARGS=()

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 2
  --sglang-mem-fraction-static 0.85
  --sglang-context-length 32768
)

# -----------------------------------------------------------------------
# Custom generate / reward hooks
# -----------------------------------------------------------------------
CUSTOM_ARGS=(
  --custom-generate-function-path math_tutor_api_server.generate
  --custom-rm-path math_tutor_api_server.reward_func
)

# -----------------------------------------------------------------------
# KC-aware OPD judge (optional)
# Set OPD_ENABLE=1 to enable; requires a separate judge model server.
# -----------------------------------------------------------------------
OPD_ARGS=()
if [ "${OPD_ENABLE}" = "1" ]; then
  export PRM_M="${PRM_M:-3}"
  OPD_ARGS=(
    --prm-enable
    --prm-num-gpus "${OPD_GPUS}"
    --prm-num-gpus-per-engine 2
    --prm-model-path "${OPD_MODEL_PATH}"
    --prm-m "${PRM_M}"
    --prm-temperature "${PRM_TEMPERATURE:-0.6}"
    --prm-max-new-tokens "${PRM_MAX_NEW_TOKENS:-4096}"
  )
  echo "[MathTutor-RL] KC-aware OPD enabled (judge model: ${OPD_MODEL_PATH})"
else
  echo "[MathTutor-RL] OPD disabled — using SymPy-only rewards"
fi

# -----------------------------------------------------------------------
# Weights & Biases (optional)
# -----------------------------------------------------------------------
USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-math_tutor_rl}
WANDB_KEY_VALUE=${WANDB_KEY:-${WANDB_API_KEY:-}}
if [ "${USE_WANDB}" = "1" ] && [ -n "${WANDB_KEY_VALUE}" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group deepseek-math-7b-math-tutor-rl
    --wandb-key "${WANDB_KEY_VALUE}"
  )
else
  WANDB_ARGS=()
fi

# -----------------------------------------------------------------------
# Ray cluster
# -----------------------------------------------------------------------
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head \
  --node-ip-address "${MASTER_ADDR}" \
  --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${REPO_ROOT}/Megatron-LM/:${SCRIPT_DIR}:${SLIME_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"FLASHINFER_WORKSPACE_BASE\": \"${FLASHINFER_WORKSPACE_BASE}\",
    \"MATH_TUTOR_MEMORY_DIR\": \"${MATH_TUTOR_MEMORY_DIR}\",
    \"MATH_TUTOR_RECORD_ENABLED\": \"${MATH_TUTOR_RECORD_ENABLED}\",
    \"MATH_TUTOR_RECORD_FILE\": \"${MATH_TUTOR_RECORD_FILE}\",
    \"SERVED_MODEL_NAME\": \"${SERVED_MODEL_NAME}\",
    \"HOST\": \"${HOST}\",
    \"PORT\": \"${PORT}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 "${SLIME_ROOT}/train_async.py" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${ACTOR_GPUS}" \
  --rollout-num-gpus "${ROLLOUT_GPUS}" \
  --num-gpus-per-node "${NUM_GPUS}" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${CUSTOM_ARGS[@]}" \
  "${OPD_ARGS[@]}"

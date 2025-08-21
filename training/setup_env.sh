#!/bin/bash
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML_PATH="$(dirname "$script_dir")"

echo "ML directory: $ML_PATH"

BASE_PATH="/data/stefan"
echo "Base path (datasets, experiments): $BASE_PATH"

export PYTHONPATH="$ML_PATH"

echo "Setting up cache directories in $BASE_PATH/.cache"
export HF_HOME=${BASE_PATH}/.cache/huggingface_cache
export HF_DATASETS_CACHE=${BASE_PATH}/.cache/huggingface_cache
export VLLM_CACHE_ROOT=${BASE_PATH}/.cache/vllm_cache
export TRITON_CACHE_DIR=${BASE_PATH}/.cache/triton_cache/
export UV_CACHE_DIR=${BASE_PATH}/.cache/uv_cache
export WANDB_CACHE_DIR=${BASE_PATH}/.cache/wandb
export WANDB_DATA_DIR=${BASE_PATH}/.cache
export WANDB_ARTIFACT_LOCATION=${BASE_PATH}/.cache/wandb/artifacts
export WANDB_ARTIFACT_DIR=${BASE_PATH}/.cache/wandb/artifacts
export WANDB_CONFIG_DIR=${BASE_PATH}/.cache/wandb/config
# export TMPDIR=${BASE_PATH}/.cache/tmp
export XDG_CACHE_HOME=${BASE_PATH}/.cache

cd ${ML_PATH}/training
source .venv/bin/activate


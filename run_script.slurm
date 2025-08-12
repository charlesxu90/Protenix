#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=64G
#SBATCH --constraint="a100"
#SBATCH --time=4:00:00
#SBATCH --partition=batch
#SBATCH --output=log-%x-%j.out
#SBATCH --error=log-%x-%j.out

module load cuda/12.1
# source ~/miniconda3/etc/profile.d/conda.sh
source ~/.bashrc
source /ibex/user/fengc0a/soft/miniconda3/etc/profile.d/conda.sh
PROTENIX_DIR="/ibex/project/c2108/xux/software/protenix"
conda activate $PROTENIX_DIR/env

export LAYERNORM_TYPE=fast_layernorm
export USE_DEEPSPEED_EVO_ATTENTION=true
export CUTLASS_PATH="$PROTENIX_DIR/cutlass"

input_json_path=$1 # Example: "./examples/example.json"

# python3 $PROTENIX_DIR/Protenix/runner/inference.py \
# --model_name "protenix_base_default_v0.5.0"\
# --seeds 101 \
# --dump_dir "${input_json_path}-output" \
# --input_json_path ${input_json_path} \
# --model.N_cycle 10 \
# --sample_diffusion.N_sample 5 \
# --sample_diffusion.N_step 200

protenix predict --input  ${input_json_path} --out_dir "${input_json_path}-output" --seeds 101 --model_name "protenix_base_default_v0.5.0"
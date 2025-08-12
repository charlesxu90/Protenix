#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=64G
#SBATCH --constraint="v100"
#SBATCH --time=24:00:00
#SBATCH --partition=batch
#SBATCH --output=log-%x-%j.out
#SBATCH --error=log-%x-%j.out

module load cuda/12.1
# source ~/miniconda3/etc/profile.d/conda.sh
source /ibex/user/fengc0a/soft/miniconda3/etc/profile.d/conda.sh
PROTENIX_DIR="/ibex/project/c2108/xux/software/protenix"
conda activate $PROTENIX_DIR/env

export LAYERNORM_TYPE=fast_layernorm
export USE_DEEPSPEED_EVO_ATTENTION=true
export CUTLASS_PATH="$PROTENIX_DIR/cutlass"

N_sample=5
N_step=200
N_cycle=10
seed=101

input_json_path=$1 # Example: "./examples/example.json"
dump_dir="${input_json_path}-output"
model_name="protenix_base_default_v0.5.0"

python3 $PROTENIX_DIR/protenix/runner/inference.py \
--model_name ${model_name} \
--seeds ${seed} \
--dump_dir ${dump_dir} \
--input_json_path ${input_json_path} \
--model.N_cycle ${N_cycle} \
--sample_diffusion.N_sample ${N_sample} \
--sample_diffusion.N_step ${N_step}
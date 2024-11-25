#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --job-name="vcl-nstepkl"
#SBATCH --output=/users/lucelo/logs/slurm-%j.out
#SBATCH --error=/users/lucelo/logs/slurm-%j.err

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs
export XDG_CACHE_HOME=/scratch-ssd/oatml/

export TMPDIR=/scratch-ssd/${USER}/tmp
mkdir -p $TMPDIR
BUILD_DIR=/scratch-ssd/${USER}/conda_envs/pip-build

/scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f ~/vcl-nstepkl/environment_cloud.yml
source /scratch-ssd/oatml/miniconda3/bin/activate vcl-nstepkl

echo $TMPDIR

nvidia-smi

# Function to generate a random seed
generate_seed() {
  echo $((RANDOM % 10000))
}

cd ~/vcl-nstepkl


python3 -u run_hypersearch_imagenet_ucl.py

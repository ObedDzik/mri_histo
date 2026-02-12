#!/bin/bash
#SBATCH -J isup-proj-triplet
#SBATCH -A aip-medilab
#SBATCH --gres=gpu:l40s:1
#SBATCH -t 00:50:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -D /project/aip-medilab/shared/picai
#SBATCH -o /home/obed/projects/aip-medilab/obed/medproj/mri_histo/results/testmri_histo_train_encoder-%j.out


source /project/aip-medilab/obed/medproj/mri_histo/mri_env/bin/activate
export DINOV3_LIBRARY_PATH=/home/obed/projects/aip-medilab/obed/medproj/dinov3
export DINOV3_CHECKPOINTS_PATH=/datasets/exactvu_pca/checkpoint_store
export CHECKPOINT=/scratch/obed
export JOB_ID=$SLURM_JOB_ID

srun python /home/obed/projects/aip-medilab/obed/medproj/mri_histo/train.py -c /home/obed/projects/aip-medilab/obed/medproj/mri_histo/cfg.yaml

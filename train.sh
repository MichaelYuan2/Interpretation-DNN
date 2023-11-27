#! /bin/bash
#SBATCH --verbose
#SBATCH --job-name compet_train
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # select which email types will be sent
#SBATCH --mail-user=yx2433@nyu.edu


#SBATCH --array=1
#SBATCH --output=sbl_%A_%a.out # %A is SLURM_ARRAY_JOB_ID, %a is SLURM_ARRAY_TASK_ID
#SBATCH --error=sbl_%A_%a.err

echo === $(date)
echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo ${SLURM_ARRAY_TASK_ID}

nvidia-smi
singularity exec --nv \
	    --overlay /scratch/yx2433/base/overlay-25GB-500K.ext3:ro \
	    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif\
	    /bin/bash -c "source /ext3/env.sh; python train.py"
#!/bin/bash
#SBATCH -o sbatchlogs/job.%j.out
#SBATCH --partition=shenlong
#SBATCH --nodes=1             
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1    
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00    
#SBATCH --mem=16G     

# conda activate KITTI360
# srun --partition=shenlong --time=24:00:00 --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=4 --mem=16000 --pty /bin/bash
# python run.py
python run_LS.py
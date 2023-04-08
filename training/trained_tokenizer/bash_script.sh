#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=100:0:0
#SBATCH --mail-user=nku618@uregina.ca
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:v100l:1


nvidia-smi
cd ~
source Torch/bin/activate
cd ~/projects/def-nshahria/nku618/NCT_Tickets/training
python eval_baseline.py

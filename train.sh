#!/bin/bash
#SBATCH -p H100,H200,H200-SDS
#SBATCH --job-name="CIFAR10-DOCC"
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

#SBATCH --container-image=/netscratch/mukhtar/Pipeline/nvcr.io_nvidia_pytorch_25.02-py3.sqsh
#SBATCH --container-workdir=/netscratch/mukhtar/DOCC
#SBATCH --container-mounts=/netscratch:/netscratch
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

python -m src.supervised
# python -m src.simdocc

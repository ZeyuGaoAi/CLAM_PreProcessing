#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time 8:0:0
#SBATCH --mem-per-cpu=3850
#SBATCH --account=su123
#SBATCH -o ./logs/slurm.%N.%j.out # STDOUT
#SBATCH -e ./logs/slurm.%N.%j.err # STDERR

module purge
module load GCCcore/11.3.0 Python/3.10.4
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0
module load GCCcore/11.3.0 OpenSlide/3.4.1-largefiles

source /home/z/zeyugao/pyvenv/conch/bin/activate

python create_patches_fp.py --source /home/shared/su123/TCGA_ORI/TCGA-OV/ \
       --save_dir /home/shared/su123/TCGA_Embed/TCGA-OV/clam_gen_1024   \
       --step_size 1024 --patch_size 1024 --seg --patch --stitch --patch_level 0 --preset tcga.csv &

python create_patches_fp.py --source /home/shared/su123/TCGA_ORI/TCGA-OV/ \
       --save_dir /home/shared/su123/TCGA_Embed/TCGA-OV/clam_gen_512   \
       --step_size 512 --patch_size 512 --seg --patch --stitch --patch_level 0 --preset tcga.csv &

python create_patches_fp.py --source /home/shared/su123/TCGA_ORI/TCGA-OV/ \
       --save_dir /home/shared/su123/TCGA_Embed/TCGA-OV/clam_gen_256   \
       --step_size 256 --patch_size 256 --seg --patch --stitch --patch_level 0 --preset tcga.csv &

wait
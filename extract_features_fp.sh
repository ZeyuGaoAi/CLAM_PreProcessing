#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --tasks-per-node 1
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --partition=gpu
#SBATCH --time 72:0:0
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

export CONCH_CKPT_PATH="/home/z/zeyugao/PreModel/conch/pytorch_model.bin"
export UNI_CKPT_PATH="/home/z/zeyugao/PreModel/uni/pytorch_model.bin"

# uni 224, conch_v1_5 448, h_optimus_0 224, virchow2 224, gigapath 224, hibou_l 224

python extract_features_fp.py --data_h5_dir /home/shared/su123/TCGA_Embed/TCGA-RCC/clam_gen_512/ \
    --anno_dir /home/z/zeyugao/dataset/WSIData/TCGA-RCC/annotation/ \
    --data_slide_dir /home/shared/su123/TCGA_ORI/TCGA-RCC/ \
    --csv_path /home/shared/su123/TCGA_Embed/TCGA-RCC/clam_gen_512/process_list_autogen.csv --model_name virchow2\
    --feat_dir /home/shared/su123/TCGA_Embed/TCGA-RCC/clam_gen_512/virchow2/ \
    --suffix "_0_512" --patch_size 512 &

python extract_features_fp.py --data_h5_dir /home/shared/su123/TCGA_Embed/TCGA-RCC/clam_gen_512/ \
    --anno_dir /home/z/zeyugao/dataset/WSIData/TCGA-RCC/annotation/ \
    --data_slide_dir /home/shared/su123/TCGA_ORI/TCGA-RCC/ \
    --csv_path /home/shared/su123/TCGA_Embed/TCGA-RCC/clam_gen_512/process_list_autogen.csv --model_name gigapath\
    --feat_dir /home/shared/su123/TCGA_Embed/TCGA-RCC/clam_gen_512/gigapath/ \
    --suffix "_0_512" --patch_size 512 & # --target_patch_size 448

python extract_features_fp.py --data_h5_dir /home/shared/su123/TCGA_Embed/TCGA-RCC/clam_gen_512/ \
    --anno_dir /home/z/zeyugao/dataset/WSIData/TCGA-RCC/annotation/ \
    --data_slide_dir /home/shared/su123/TCGA_ORI/TCGA-RCC/ \
    --csv_path /home/shared/su123/TCGA_Embed/TCGA-RCC/clam_gen_512/process_list_autogen.csv --model_name hibou_l\
    --feat_dir /home/shared/su123/TCGA_Embed/TCGA-RCC/clam_gen_512/hibou_l/ \
    --suffix "_0_512" --patch_size 512 &

# python extract_features_fp.py --data_h5_dir /home/z/zeyugao/dataset/WSIData/TCGA-RCC/clam_gen_512/ \
#     --anno_dir /home/z/zeyugao/dataset/WSIData/TCGA-RCC/annotation/ \
#     --data_slide_dir /home/shared/su123/TCGA_ORI/TCGA-RCC/ \
#     --csv_path /home/z/zeyugao/dataset/WSIData/TCGA-RCC/clam_gen_512/process_list_autogen.csv --model_name uni_v1\
#     --feat_dir /home/z/zeyugao/dataset/WSIData/TCGA-RCC/clam_gen_512/uni/ --suffix "_0_512" --patch_size 512 &

# python extract_features_fp.py --data_h5_dir /home/z/zeyugao/dataset/WSIData/TCGA-RCC/clam_gen_512/ \
#     --anno_dir /home/z/zeyugao/dataset/WSIData/TCGA-RCC/annotation/ \
#     --data_slide_dir /home/shared/su123/TCGA_ORI/TCGA-RCC/ \
#     --csv_path /home/z/zeyugao/dataset/WSIData/TCGA-RCC/clam_gen_512/process_list_autogen.csv --model_name conch_v1_5\
#     --feat_dir /home/z/zeyugao/dataset/WSIData/TCGA-RCC/clam_gen_512/conch_v1_5/ \
#     --suffix "_0_512" --patch_size 512 --target_patch_size 448 &

# python extract_features_fp.py --data_h5_dir /home/z/zeyugao/dataset/WSIData/TCGA-RCC/clam_gen/ \
#     --anno_dir /home/z/zeyugao/dataset/WSIData/TCGA-RCC/annotation/ --data_slide_dir /home/shared/su123/TCGA_ORI/TCGA-RCC/ \
#     --csv_path /home/z/zeyugao/dataset/WSIData/TCGA-RCC/clam_gen/process_list_autogen.csv --model_name resnet50_trunc\
#     --feat_dir /home/z/zeyugao/dataset/WSIData/TCGA-RCC/clam_gen/resnet50/ --suffix "_1_512" --patch_size 2048 &

# python extract_features_fp.py --data_h5_dir /home/z/zeyugao/dataset/WSIData/TCGA-LUNG/clam_gen_224/ \
#     --anno_dir /home/z/zeyugao/dataset/WSIData/TCGA-LUNG/annotation/ --data_slide_dir /home/shared/su123/TCGA_ORI/TCGA-LUNG/ \
#     --csv_path /home/z/zeyugao/dataset/WSIData/TCGA-LUNG/clam_gen_224/process_list_autogen.csv --model_name dsmil_lung\
#     --feat_dir /home/z/zeyugao/dataset/WSIData/TCGA-LUNG/clam_gen_224/dsmil/ --suffix "_1_224" --patch_size 896 &

# python extract_features_fp.py --data_h5_dir /home/z/zeyugao/dataset/WSIData/Camelyon/clam_gen_224/ \
#     --anno_dir /home/z/zeyugao/dataset/WSIData/Camelyon/annotation/ --data_slide_dir /home/shared/su123/Camelyon/WSIs/ \
#     --csv_path /home/z/zeyugao/dataset/WSIData/Camelyon/clam_gen_224/process_list_autogen.csv --model_name dsmil_camel\
#     --feat_dir /home/z/zeyugao/dataset/WSIData/Camelyon/clam_gen_224/dsmil/ --slide_ext '.tif' --suffix "_1_224" --patch_size 448 &

# python extract_features_fp_patch.py --data_dir /home/shared/su123/SICAPv2/Patch/ \
#     --feat_dir /home/z/zeyugao/dataset/WSIData/SICAPv2/clam_gen/resnet50/\
#      --patch_ext '.jpg' --suffix "_0_1024" --patch_size 1024 --model_name resnet50_trunc &

# python extract_features_fp.py --data_h5_dir /home/z/zeyugao/dataset/WSIData/TCGA-STAD/clam_gen/ \
#     --anno_dir /home/z/zeyugao/dataset/WSIData/TCGA-STAD/annotation/ --data_slide_dir /home/shared/su123/TCGA_ORI/TCGA-STAD/ \
#     --csv_path /home/z/zeyugao/dataset/WSIData/TCGA-STAD/clam_gen/process_list_autogen.csv --model_name resnet50_trunc\
#     --feat_dir /home/z/zeyugao/dataset/WSIData/TCGA-STAD/clam_gen/resnet50/ --suffix "_1_512" --patch_size 2048 &

# python extract_features_fp_patch.py --data_dir /home/shared/su123/TCGA_ORI/TCGA-STAD-Patch/ \
#     --feat_dir /home/z/zeyugao/dataset/WSIData//TCGA-STAD/clam_gen/resnet50_patch/\
#      --patch_ext '.png' --suffix "_1_512" --patch_size 2048 --model_name resnet50_trunc &

wait
# CLAM_PreProcessing
CLAM-based preprocessing with more foundation models

| Version Name      | Link      |
|-------------------|-------------------|
| uni_v1            |https://huggingface.co/MahmoodLab/UNI|
| conch_v1          |https://huggingface.co/MahmoodLab/CONCH|
| conch_v1_5        |https://huggingface.co/MahmoodLab/TITAN|
| h_optimus_0       |https://huggingface.co/bioptimus/H-optimus-0|
| virchow2          |https://huggingface.co/paige-ai/Virchow2|
| gigapath          |https://huggingface.co/prov-gigapath/prov-gigapath|
| hibou_l           |https://huggingface.co/histai/hibou-L|

## Patch Tessellation
```
python create_patches_fp.py --source /home/shared/su123/TCGA_ORI/TCGA-OV/ \
       --save_dir /home/shared/su123/TCGA_Embed/TCGA-OV/clam_gen_1024   \
       --step_size 1024 --patch_size 1024 --seg --patch --stitch --patch_level 0 --preset tcga.csv
```

## Feature Extraction

set up model_name: uni_v1, conch_v1, conch_v1_5, h_optimus_0, virchow2, gigapath, hibou_l

```
python extract_features_fp.py --data_h5_dir /home/shared/su123/TCGA_Embed/TCGA-RCC/clam_gen_1024/ \
    --anno_dir /home/z/zeyugao/dataset/WSIData/TCGA-RCC/annotation/ \
    --data_slide_dir /home/shared/su123/TCGA_ORI/TCGA-RCC/ \
    --csv_path /home/shared/su123/TCGA_Embed/TCGA-RCC/clam_gen_1024/process_list_autogen.csv --model_name virchow2\
    --feat_dir /home/shared/su123/TCGA_Embed/TCGA-RCC/clam_gen_1024/virchow2/ \
    --suffix "_0_1024" --patch_size 1024 
```

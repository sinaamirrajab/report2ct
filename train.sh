# creating image embeddings using the pretrained MAISI VAE GAN model
python -m src.maisi.scripts.vlm3d_image_embedding 

# training the report conditional diffusion model for report to CT synthesis
torchrun --nproc_per_node=2 -m src.maisi.scripts.diff_model_train_vlm3D_2560_multi_text \
    --env_config ./vlm3D_work_dir/environment_maisi_diff_model_vlm3D_FI_2560_multi.json \
    --model_config ./vlm3D_work_dir/config_maisi_diff_model_vlm3D.json \
    --model_def ./vlm3D_work_dir/config_maisi_2560.json \
    --num_gpus 2 

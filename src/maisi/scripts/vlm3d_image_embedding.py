#!/usr/bin/env python3
"""
Training a 3D Diffusion Model for Generating 3D Images with Various Sizes and Spacings

Copyright (c) MONAI Consortium  
Licensed under the Apache License, Version 2.0 (the "License");  
you may not use this file except in compliance with the License.  
You may obtain a copy of the License at  
    http://www.apache.org/licenses/LICENSE-2.0  
Unless required by applicable law or agreed to in writing, software  
distributed under the License is distributed on an "AS IS" BASIS,  
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
See the License for the specific language governing permissions and  
limitations under the License.

In this script, we detail the procedure for training a 3D latent diffusion model 
to generate high-dimensional 3D medical images. Due to the potential for out-of-memory 
issues on most GPUs when generating large images (e.g., those with dimensions of 
512 x 512 x 512 or greater), we have structured the training process into two primary 
steps: 1) generating image embeddings and 2) training 3D latent diffusion models. 
The subsequent sections will demonstrate the entire process using a simulated dataset.

[Release Note (March 2025)]: We are excited to announce the new MAISI Version 
'maisi3d-rflow'. Compared with the previous version 'maisi3d-ddpm', it accelerated 
latent diffusion model inference by 33x.
"""

import copy
import os
import json
import numpy as np
import nibabel as nib
import subprocess
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import torch

from monai.apps import download_url
from monai.data import create_test_image_3d
from monai.config import print_config
from monai.transforms import LoadImage, EnsureChannelFirst, Orientation, ScaleIntensityRange
from monai.visualize import plot_2d_or_3d_image

from src.maisi.scripts.diff_model_setting import setup_logging

def main():
    """Main function to run the MAISI training pipeline."""
    
    # Setup
    print_config()
    logger = setup_logging("notebook")
    
    # Set up the MAISI version
    # Choose between 'maisi3d-ddpm' and 'maisi3d-rflow'. The differences are:
    # - The maisi version 'maisi3d-ddpm' uses basic noise scheduler DDPM. 
    #   'maisi3d-rflow' uses Rectified Flow scheduler, can be 33 times faster during inference.
    # - The maisi version 'maisi3d-ddpm' requires training images to be labeled with 
    #   body regions ("top_region_index" and "bottom_region_index"), while 'maisi3d-rflow' 
    #   does not have such requirement. In other words, it is easier to prepare training 
    #   data for 'maisi3d-rflow'.
    # - For the released model weights, 'maisi3d-rflow' can generate images with better 
    #   quality for head region and small output volumes, and comparable quality for 
    #   other cases compared with 'maisi3d-ddpm'.
    
    maisi_version = "maisi3d-rflow"
    if maisi_version == "maisi3d-ddpm":
        model_def_path = "src/maisi/configs/config_maisi3d-ddpm.json"
    elif maisi_version == "maisi3d-rflow":
        model_def_path = "src/maisi/configs/config_maisi3d-rflow_vlm3D.json"
    else:
        raise ValueError(f"maisi_version has to be chosen from ['maisi3d-ddpm', 'maisi3d-rflow'], yet got {maisi_version}.")
    
    with open(model_def_path, "r") as f:
        model_def = json.load(f)
    include_body_region = model_def["include_body_region"]
    logger.info(f"MAISI version is {maisi_version}, whether to use body_region is {include_body_region}")
    
    print("Model definition:", model_def)
    
    # Simulate a special dataset
    # It is widely recognized that training AI models is a time-intensive process. 
    # In this instance, we will simulate a small dataset and conduct training over 
    # multiple epochs. While the performance may not reach optimal levels due to 
    # the abbreviated training duration, the entire pipeline will be completed within minutes.
    # sim_datalist provides the information of the simulated datasets. It lists 2 training images.
    
    csv_data_dir = './csv_data'
    train_df = pd.read_csv(os.path.join(csv_data_dir, 'train_reports_patients.csv'))
    
    # sim_datalist = {"training": [{"image": train_df["nifti_path"].iloc[0]}, {"image": train_df["nifti_path"].iloc[10]}]}

    sim_datalist = {"training": []}
    for i in range(min(21000, len(train_df))):
        row = train_df.iloc[i]
        entry = {
            "image": row["nifti_path"],
            # "embedding": row["Impressions_EN_embeddings"],
            "volume_name": row["patient_name"]
        }
        sim_datalist["training"].append(entry)

    print(f"sim_datalist: {sim_datalist}")
    print(f"Number of training images: {len(sim_datalist['training'])}")
    del train_df  # Free memory by deleting the DataFrame
    # Generate simulated images
    # Now we can use MONAI create_test_image_3d and nib.Nifti1Image functions to 
    # generate the 3D simulated images under the work_dir.
    
    # work_dir = "./temp_work_dir"
    work_dir = "./vlm3D_work_dir"
    # work_dir = '/mnt/data12/VLM3D_workdir'
    if not os.path.isdir(work_dir):
        os.makedirs(work_dir)
    
    dataroot_dir = os.path.join(work_dir, "sim_dataroot")
    if not os.path.isdir(dataroot_dir):
        os.makedirs(dataroot_dir)
    
    # Save the updated sim_datalist
    datalist_file = os.path.join(work_dir, "sim_datalist.json")
    print(f"Saving simulated datalist to {datalist_file}")
    with open(datalist_file, "w") as f:
        json.dump(sim_datalist, f)
    
    logger.info("Generated simulated images with updated file paths.")
    
    # Set up directories and configurations
    # To optimize the demonstration for time efficiency, we have adjusted the training 
    # epochs to 2. Additionally, we modified the num_splits parameter in AutoencoderKlMaisi 
    # from its default value of 16 to 4. This adjustment reduces the spatial splitting 
    # of feature maps in convolutions, which is particularly beneficial given the smaller 
    # input size. (This change helps convert convolutions to a for-loop based approach, 
    # thereby conserving GPU memory resources.)
    
    env_config_path = "src/maisi/configs/environment_maisi_diff_model_vlm3D.json"
    model_config_path = "src/maisi/configs/config_maisi_diff_model_vlm3D.json"
    
    # Load environment configuration, model configuration and model definition
    with open(env_config_path, "r") as f:
        env_config = json.load(f)
    
    with open(model_config_path, "r") as f:
        model_config = json.load(f)
    
    env_config_out = copy.deepcopy(env_config)
    model_config_out = copy.deepcopy(model_config)
    model_def_out = copy.deepcopy(model_def)
    
    # Set up directories based on configurations
    env_config_out["data_base_dir"] = dataroot_dir
    env_config_out["embedding_base_dir"] = os.path.join(work_dir, "embeddings")
    env_config_out["json_data_list"] = datalist_file
    env_config_out["model_dir"] = os.path.join(work_dir, "models")
    env_config_out["output_dir"] = os.path.join(work_dir, "predictions")
    trained_autoencoder_path = os.path.join(work_dir, "models/autoencoder_epoch273.pt")
    env_config_out["trained_autoencoder_path"] = trained_autoencoder_path
    trained_autoencoder_path_url = (
        "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/"
        "model_zoo/model_maisi_autoencoder_epoch273_alternative.pt"
    )
    if not os.path.exists(trained_autoencoder_path):
        download_url(url=trained_autoencoder_path_url, filepath=trained_autoencoder_path)
    
    # Create necessary directories
    os.makedirs(env_config_out["embedding_base_dir"], exist_ok=True)
    os.makedirs(env_config_out["model_dir"], exist_ok=True)
    os.makedirs(env_config_out["output_dir"], exist_ok=True)
    
    env_config_filepath = os.path.join(work_dir, "environment_maisi_diff_model_vlm3D.json")
    with open(env_config_filepath, "w") as f:
        json.dump(env_config_out, f, sort_keys=True, indent=4)
    
    # Update model configuration for demo
    max_epochs = 100
    batch_size = 2  # Adjusted for demonstration purposes
    model_config_out["diffusion_unet_train"]["n_epochs"] = max_epochs
    model_config_out["diffusion_unet_train"]["batch_size"] = batch_size

    
    model_config_filepath = os.path.join(work_dir, "config_maisi_diff_model_vlm3D.json")
    with open(model_config_filepath, "w") as f:
        json.dump(model_config_out, f, sort_keys=True, indent=4)
    
    model_def_filepath = os.path.join(work_dir, "config_maisi.json")
    with open(model_def_filepath, "w") as f:
        json.dump(model_def_out, f, sort_keys=True, indent=4)
    
    # Print files and folders under work_dir
    logger.info(f"files and folders under work_dir: {os.listdir(work_dir)}.")
    
    # Adjust based on the number of GPUs you want to use
    num_gpus = 1
    logger.info(f"number of GPUs: {num_gpus}.")
    
    print(f"config_maisi_diff_model: {model_config_filepath}")
    print(f"environment_maisi_diff_model: {env_config_filepath}")
    print(f"config_maisi: {model_def_filepath}")
    
    # Step 1: Create Training Data
    # To train the latent diffusion model, we first store the latent features produced 
    # by the autoencoder's encoder in local storage. This allows the latent diffusion 
    # model to directly utilize these features, thereby conserving both time and GPU 
    # memory during the training process. Additionally, we have provided the script 
    # for multi-GPU processing to save latent features from all training images, 
    # significantly accelerating the creation of the entire training set.
    # The diffusion model utilizes a U-shaped convolutional neural network architecture, 
    # requiring matching input and output dimensions. Therefore, it is advisable to 
    # resample the input image dimensions to be multiples of 2 for compatibility. 
    # In this case, we have chosen dimensions that are multiples of 128.
    
    logger.info("Creating training data...")
    
    # Define the arguments for torchrun
    module = "src.maisi.scripts.diff_model_create_training_data_vlm3D_all"
    module_args = [
        "--env_config",
        env_config_filepath,
        "--model_config",
        model_config_filepath,
        "--model_def",
        model_def_filepath,
        "--num_gpus",
        str(num_gpus),
    ]
    
    # Add metadata path for HU rescaling if available no need to rescale HU
    metadata_path = os.path.join(csv_data_dir, 'train_reports_20250710.csv')
    if os.path.exists(metadata_path):
        module_args.extend(["--metadata_path", metadata_path])
        logger.info(f"Added metadata path for HU rescaling: {metadata_path}")
    else:
        logger.warning(f"Metadata file not found at {metadata_path}, HU rescaling will be skipped")
    
    gpu_ids = 1
    run_torchrun(module, module_args, num_gpus=num_gpus, gpu_ids=gpu_ids)
    
    # Create .json files for embedding files
    # The diffusion model necessitates additional input attributes, including output 
    # dimension, output spacing, and top/bottom body region. These dimensions and 
    # spacing can be extracted from the header information of the training images. 
    # The top and bottom body region inputs can be determined through manual examination 
    # or by utilizing segmentation masks from tools such as TotalSegmentator or MONAI VISTA. 
    # The body regions are formatted as 4-dimensional one-hot vectors: the head and neck 
    # region is represented by [1,0,0,0], the chest region by [0,1,0,0], the abdomen 
    # region by [0,0,1,0], and the lower body region (below the abdomen) by [0,0,0,1]. 
    # The additional input attributes are saved in a separate .json file. In the following 
    # example, we assume that the images cover the chest and abdomen regions.
    
    # folder_path = env_config_out["embedding_base_dir"]
    # gz_files = list_gz_files(folder_path)
    # create_json_files(gz_files, include_body_region, logger, sim_datalist)
    
    # logger.info("Completed creating .json files for all embedding files.")
    
    # Check the config files
    print(env_config_filepath)
    print(model_config_filepath)
    print(model_def_filepath)
    
    # Step 2: Train the Model
    # After all latent features have been created, we will initiate the multi-GPU 
    # script to train the latent diffusion model.
    # The image generation process utilizes the DDPM scheduler with 1,000 iterative steps. 
    # The diffusion model is optimized using L1 loss and a decayed learning rate scheduler. 
    # The batch size for this process is set to 1.
    # Please be aware that using the H100 GPU may occasionally result in random 
    # segmentation faults. To avoid this issue, you can disable AMP by setting 
    # the --no_amp flag.
    

def run_torchrun(module, module_args, num_gpus=1, gpu_ids=None):
    """Run a module using torchrun with specified arguments."""
    # Define the arguments for torchrun
    num_nodes = 1

    # Build the torchrun command
    torchrun_command = [
        "torchrun",
        "--nproc_per_node",
        str(num_gpus),
        "--nnodes",
        str(num_nodes),
        "-m",
        module,
    ] + module_args

    # Set the environment variables
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    
    # Set CUDA_VISIBLE_DEVICES to specify which GPU(s) to use
    if gpu_ids is not None:
        if isinstance(gpu_ids, int):
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)
        elif isinstance(gpu_ids, (list, tuple)):
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)

    # Execute the command with real-time output
    process = subprocess.Popen(torchrun_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                              text=True, env=env, bufsize=1, universal_newlines=True)

    # Print the output in real-time
    try:
        for line in iter(process.stdout.readline, ''):
            if line:
                # Print immediately and flush to ensure real-time display
                print(line.rstrip(), flush=True)
            if process.poll() is not None:
                break
        
        # Wait for process to complete
        process.wait()
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user...")
        process.terminate()
        process.wait()
    except Exception as e:
        print(f"An error occurred: {e}")
        process.terminate()
        process.wait()
    return

def list_gz_files(folder_path):
    """List all .gz files in the folder and its subfolders."""
    gz_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".gz"):
                gz_files.append(os.path.join(root, file))
    return gz_files

def create_json_files(gz_files, include_body_region, logger, sim_datalist):
    """Create .json files for each .gz file with the specified keys and values, and add impression embeddings."""
    for gz_file in gz_files:
        # Load the NIfTI image
        img = nib.load(gz_file)

        # Get the dimensions and spacing
        dimensions = img.shape
        dimensions = dimensions[:3]
        spacing = img.header.get_zooms()[:3]
        spacing = spacing[:3]
        spacing = [float(_item) for _item in spacing]

        # Create the dictionary with the specified keys and values
        data = {"dim": dimensions, "spacing": spacing}
        if include_body_region:
            # The region can be selected from one of four regions from top to bottom.
            # [1,0,0,0] is the head and neck, [0,1,0,0] is the chest region, [0,0,1,0]
            # is the abdomen region, and [0,0,0,1] is the lower body region.
            data["top_region_index"] = [0, 1, 0, 0]  # chest region
            data["bottom_region_index"] = [0, 0, 1, 0]  # abdomen region

        # Add Impressions_EN_embeddings from sim_datalist by matching VolumeName
        volume_name = os.path.basename(gz_file).replace("_emb", "")
        embedding_found = False
        for entry in sim_datalist["training"]:
            if entry["volume_name"] == volume_name:
                embedding = entry["embedding"]
                if isinstance(embedding, str):
                    try:
                        import ast
                        embedding = ast.literal_eval(embedding)
                    except Exception:
                        pass
                data["Impressions_EN_embeddings"] = embedding
                embedding_found = True
                break
        
        if not embedding_found:
            logger.warning(f"No matching VolumeName for {volume_name} in sim_datalist.")

        # logger.info(f"data: {data}.")

        # Create the .json filename
        json_filename = gz_file + ".json"

        # Write the dictionary to the .json file
        with open(json_filename, "w") as json_file:
            json.dump(data, json_file, indent=4)
        logger.info(f"Save json file to {json_filename}")

def visualize_results(work_dir):
    """Visualize the generated results."""
    # Path to your NIfTI file
    predictions_dir = os.path.join(work_dir, "predictions")
    if not os.path.exists(predictions_dir):
        print(f"Predictions directory {predictions_dir} does not exist.")
        return
    
    # Find the first .nii.gz file in predictions directory
    nifti_files = [f for f in os.listdir(predictions_dir) if f.endswith('.nii.gz')]
    if not nifti_files:
        print(f"No .nii.gz files found in {predictions_dir}")
        return
    
    nifti_path = os.path.join(predictions_dir, nifti_files[0])
    print(f"Visualizing: {nifti_path}")
    
    # Load the image with proper transforms
    transforms = [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Orientation(axcodes="RAS")
    ]
    nib_image = nib.load(nifti_path)
    
    # Apply transforms
    image = transforms[0](nifti_path)
    image = transforms[1](image)
    image = transforms[2](image)
    
    # Optionally adjust window level for better visualization of CT data
    # Typical window for CT: window level = 40, window width = 400 (for soft tissue)
    image_windowed = ScaleIntensityRange(
        a_min=-1000,  # Lower bound of windowing range
        a_max=1000,   # Upper bound of windowing range
        b_min=0, 
        b_max=1,
        clip=True
    )(image)
    
    # Get the middle slices for each dimension
    slice_idx = [dim // 2 for dim in image.shape[1:]]
    
    # Create a figure with 3 subplots for axial, sagittal, and coronal views
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display axial view (view from top)
    axes[0].imshow(image_windowed[0, :, :, slice_idx[2]].numpy(), cmap="gray")
    axes[0].set_title(f"Axial Slice {slice_idx[2]}")
    axes[0].axis('off')
    
    # Display sagittal view (view from side)
    axes[1].imshow(image_windowed[0, :, slice_idx[1], :].numpy(), cmap="gray")
    axes[1].set_title(f"Sagittal Slice {slice_idx[1]}")
    axes[1].axis('off')
    
    # Display coronal view (view from front)
    axes[2].imshow(image_windowed[0, slice_idx[0], :, :].numpy(), cmap="gray")
    axes[2].set_title(f"Coronal Slice {slice_idx[0]}")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print image information
    print(f"Image shape: {image.shape}")
    print(f"Image data type: {image.dtype}")
    print(f"Value range: [{torch.min(image).item()}, {torch.max(image).item()}]")
    print(f"Image spacing: {nib_image.header.get_zooms()}")

if __name__ == "__main__":
    main()


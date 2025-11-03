# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from tqdm import tqdm
import ast
import monai
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
from monai.transforms import Compose, MapTransform
from monai.utils import set_determinism
from torch.amp import autocast  # Explicit import recommended
from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .utils import define_instance

# Set the random seed for reproducibility
set_determinism(seed=0)


class RescaleHUd(MapTransform):
    """
    Custom MONAI transform to apply rescale slope and intercept for HU conversion.
    
    This transform applies the formula: HU = pixel_value * rescale_slope + rescale_intercept
    """
    
    def __init__(self, keys, metadata_df=None, allow_missing_keys=False):
        """
        Args:
            keys: Keys of the corresponding items to be transformed.
            metadata_df: DataFrame containing metadata with RescaleSlope and RescaleIntercept columns.
            allow_missing_keys: Whether to allow missing keys.
        """
        super().__init__(keys, allow_missing_keys)
        self.metadata_df = metadata_df
    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if self.metadata_df is not None:
                # Extract volume name from the image path
                image_path = d[key].meta.get('filename_or_obj', '')
                volume_name = os.path.basename(image_path)
                
                # Find matching row in metadata
                matching_rows = self.metadata_df[self.metadata_df['VolumeName'] == volume_name]
                
                if not matching_rows.empty:
                    row = matching_rows.iloc[0]
                    rescale_slope = float(row['RescaleSlope'])
                    rescale_intercept = float(row['RescaleIntercept'])
                    
                    # Apply rescaling: HU = pixel_value * slope + intercept
                    d[key] = d[key] * rescale_slope + rescale_intercept
                    print(f"Applied HU rescaling to {volume_name}: slope={rescale_slope}, intercept={rescale_intercept}")
                else:
                    # Default values if no metadata found
                    print(f"Warning: No metadata found for {volume_name}, using default rescale values (slope=1.0, intercept=0.0)")
                    # Keep original values (equivalent to slope=1.0, intercept=0.0)
            else:
                # If no metadata provided, keep original values
                print(f"No metadata provided, skipping HU rescaling")
                
        return d


def create_transforms(dim: tuple = None, metadata_df=None) -> Compose:
    """
    Create a set of MONAI transforms for preprocessing.

    Args:
        dim (tuple, optional): New dimensions for resizing. Defaults to None.
        metadata_df (pd.DataFrame, optional): DataFrame containing metadata for HU rescaling.

    Returns:
        Compose: Composed MONAI transforms.
    """
    if dim:
        transforms_list = [
            monai.transforms.LoadImaged(keys="image"),
            monai.transforms.EnsureChannelFirstd(keys="image"),
            monai.transforms.Orientationd(keys="image", axcodes="RAS"),
            monai.transforms.EnsureTyped(keys="image", dtype=torch.float32),
        ]
        
        # Add HU rescaling transform if metadata is provided
        if metadata_df is not None:
            transforms_list.append(RescaleHUd(keys="image", metadata_df=metadata_df))
        
        # Add clipping and resizing transforms
        transforms_list.extend([
            monai.transforms.ScaleIntensityRanged(
                keys="image", a_min=-1000, a_max=1000, b_min=0, b_max=1, clip=True
            ),
            monai.transforms.Resized(keys="image", spatial_size=dim, mode="trilinear"),
        ])
        
        return Compose(transforms_list)
    else:
        return Compose(
            [
                monai.transforms.LoadImaged(keys="image"),
                monai.transforms.EnsureChannelFirstd(keys="image"),
                monai.transforms.Orientationd(keys="image", axcodes="RAS"),
            ]
        )


def round_number(number: int, base_number: int = 128) -> int:
    """
    Round the number to the nearest multiple of the base number, with a minimum value of the base number.

    Args:
        number (int): Number to be rounded.
        base_number (int): Number to be common divisor.

    Returns:
        int: Rounded number.
    """
    new_number = max(round(float(number) / float(base_number)), 1.0) * float(base_number)
    return int(new_number)


def save_png_slices(image_array, output_path, title_prefix="", interval=20, num_slices=10):
    """
    Save PNG images showing multiple slices in one row.
    
    Args:
        image_array (np.ndarray): 3D image array (H, W, D)
        output_path (str): Path to save the PNG file
        title_prefix (str): Prefix for the title
        interval (int): Interval between slices
        num_slices (int): Number of slices to show
    """
    try:
        # Ensure we have a 3D array
        if len(image_array.shape) == 4:
            image_array = image_array.squeeze()
        
        depth = image_array.shape[2]
        
        # Calculate slice indices with the specified interval
        start_slice = max(0, (depth - (num_slices - 1) * interval) // 2)
        slice_indices = [min(start_slice + i * interval, depth - 1) for i in range(num_slices)]
        
        # Create figure with subplots in one row
        fig, axes = plt.subplots(1, num_slices, figsize=(20, 4))
        if num_slices == 1:
            axes = [axes]
        
        for i, slice_idx in enumerate(slice_indices):
            slice_img = image_array[:, :, slice_idx]
            
            # Normalize for display
            slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)
            
            axes[i].imshow(slice_img, cmap='gray')
            axes[i].set_title(f'Slice {slice_idx}')
            axes[i].axis('off')
        
        plt.suptitle(f'{title_prefix} - Slices (interval={interval})')
        plt.tight_layout()
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved PNG slices to: {output_path}")
        
    except Exception as e:
        print(f"Error saving PNG slices to {output_path}: {e}")


def load_filenames(data_list_path: str) -> list:
    """
    Load filenames from the JSON data list.

    Args:
        data_list_path (str): Path to the JSON data list file.

    Returns:
        list: List of filenames.
    """
    with open(data_list_path, "r") as file:
        json_data = json.load(file)
    filenames_raw = json_data["training"]
    return [_item["image"] for _item in filenames_raw]


def load_metadata(metadata_path: str) -> pd.DataFrame:
    """
    Load metadata CSV file containing RescaleSlope and RescaleIntercept information.
    
    Args:
        metadata_path (str): Path to the metadata CSV file.
        
    Returns:
        pd.DataFrame: DataFrame containing metadata.
    """
    if os.path.exists(metadata_path):
        try:
            metadata_df = pd.read_csv(metadata_path)
            print(f"Loaded metadata from {metadata_path}")
            return metadata_df
        except Exception as e:
            print(f"Error loading metadata from {metadata_path}: {e}")
            return None
    else:
        print(f"Metadata file not found at {metadata_path}")
        return None


def process_file(
    filepath: str,
    args: argparse.Namespace,
    autoencoder: torch.nn.Module,
    device: torch.device,
    plain_transforms: Compose,
    new_transforms: Compose,
    logger: logging.Logger,
) -> None:
    """
    Process a single file to create training data.

    Args:
        filepath (str): Path to the file to be processed.
        args (argparse.Namespace): Configuration arguments.
        autoencoder (torch.nn.Module): Autoencoder model.
        device (torch.device): Device to process the file on.
        plain_transforms (Compose): Plain transforms.
        new_transforms (Compose): New transforms.
        logger (logging.Logger): Logger for logging information.
    """
    out_filename_base = os.path.basename(filepath).replace(".gz", "").replace(".nii", "")
    
    out_filename_base = os.path.join(args.embedding_base_dir, out_filename_base)
    
    out_filename = out_filename_base + "_emb.nii.gz"
    print(f"\n out_filename_base: {out_filename}")

    if os.path.isfile(out_filename):
        return

    test_data = {"image": os.path.join(args.data_base_dir, filepath)}
    transformed_data = plain_transforms(test_data)
    nda = transformed_data["image"]

    dim = [int(nda.meta["dim"][_i]) for _i in range(1, 4)]
    spacing = [float(nda.meta["pixdim"][_i]) for _i in range(1, 4)]

    logger.info(f"old dim: {dim}, old spacing: {spacing}")

    new_data = new_transforms(test_data)
    nda_image = new_data["image"]

    new_affine = nda_image.meta["affine"].numpy()
    nda_image = nda_image.numpy().squeeze()

    logger.info(f"new dim: {nda_image.shape}, new affine: {new_affine}")
    logger.info(f"range of nda_image: {nda_image.min()} - {nda_image.max()}")

    try:
        out_path = Path(out_filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"out_filename: {out_filename}")

        pt_nda = torch.from_numpy(nda_image).float().to(device).unsqueeze(0).unsqueeze(0)
        pt_nda = pt_nda.half()

        with torch.no_grad(), autocast('cuda', dtype=torch.float16):  # Explicitly define dtype if needed
            z = autoencoder.encode_stage_2_inputs(pt_nda)
            # Generate reconstruction for sanity check
            # recon = autoencoder.decode_stage_2_outputs(z)

        logger.info(f"z: {z.size()}, {z.dtype}")
        # logger.info(f"recon: {recon.size()}, {recon.dtype}")

        # Save embedding
        out_nda = z.squeeze().cpu().detach().numpy().transpose(1, 2, 3, 0)
        out_img = nib.Nifti1Image(np.float32(out_nda), affine=new_affine)
        nib.save(out_img, out_filename)
        print(f"Saved embedding to {out_filename}")

        # Save PNG slices for sanity check
        # volume_name = os.path.basename(filepath).replace('.nii.gz', '').replace('.nii', '')
        # png_dir = os.path.join(os.path.dirname(args.embedding_base_dir), "sanity_check_pngs")
        
        # Save original image slices
        # original_png_path = os.path.join(png_dir, f"{volume_name}.png")
        # if not os.path.exists(original_png_path):
            # save_png_slices(nda_image, original_png_path, title_prefix=f"Original {volume_name}")
        
        # Save reconstructed image slices
        # recon_array = recon.squeeze().cpu().detach().numpy()
        # recon_png_path = os.path.join(png_dir, f"{volume_name}_rec.png")
        # if not os.path.exists(recon_png_path):
            # save_png_slices(recon_array, recon_png_path, title_prefix=f"Reconstructed {volume_name}")
        
        # logger.info(f"Saved sanity check PNGs for {volume_name}")

    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}")

@torch.inference_mode()
def diff_model_create_training_data(
    env_config_path: str, model_config_path: str, model_def_path: str, num_gpus: int, metadata_path: str = None
) -> None:
    """
    Create training data for the diffusion model.

    Args:
        env_config_path (str): Path to the environment configuration file.
        model_config_path (str): Path to the model configuration file.
        model_def_path (str): Path to the model definition file.
        num_gpus (int): Number of GPUs to use.
        metadata_path (str, optional): Path to metadata CSV file containing RescaleSlope and RescaleIntercept.
    """
    args = load_config(env_config_path, model_config_path, model_def_path)
    # Add metadata_path to args if provided
    if metadata_path:
        args.metadata_path = metadata_path
    local_rank, world_size, device = initialize_distributed(num_gpus=num_gpus)
    logger = setup_logging("creating training data")
    logger.info(f"Using device {device}")

    autoencoder = define_instance(args, "autoencoder_def").to(device)
    try:
        checkpoint_autoencoder = torch.load(args.trained_autoencoder_path, weights_only=True)
        autoencoder.load_state_dict(checkpoint_autoencoder)
        autoencoder.eval()
    except Exception:
        logger.error("The trained_autoencoder_path does not exist!")

    Path(args.embedding_base_dir).mkdir(parents=True, exist_ok=True)

    filenames_raw = load_filenames(args.json_data_list)
    logger.info(f"filenames_raw: {filenames_raw}")

    # Load metadata for HU rescaling if available
    metadata_df = None
    if hasattr(args, 'metadata_path') and args.metadata_path:
        metadata_df = load_metadata(args.metadata_path)
    else:
        # Try to find metadata in common locations
        potential_metadata_paths = [
            os.path.join(os.path.dirname(args.data_base_dir), "metadata", "train_metadata.csv"),
            os.path.join(args.data_base_dir, "metadata", "train_metadata.csv"),
            os.path.join(args.data_base_dir, "train_metadata.csv"),
        ]
        for metadata_path in potential_metadata_paths:
            if os.path.exists(metadata_path):
                metadata_df = load_metadata(metadata_path)
                break

    plain_transforms = create_transforms(dim=None)

    for _iter in tqdm(range(len(filenames_raw))):
        if _iter % world_size != local_rank:
            continue

        filepath = filenames_raw[_iter]
        # new_dim = tuple(
        #     round_number(
        #         int(plain_transforms({"image": os.path.join(args.data_base_dir, filepath)})["image"].meta["dim"][_i])
        #     ) 
        #     for _i in range(1, 4)
        # )
        # new_dim = (512, 512, 256)
        new_dim = (480, 480, 256)
        new_transforms = create_transforms(new_dim, metadata_df=metadata_df)

        process_file(filepath, args, autoencoder, device, plain_transforms, new_transforms, logger)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Training Data Creation")
    parser.add_argument(
        "--env_config",
        type=str,
        default="./configs/environment_maisi_diff_model_train.json",
        help="Path to environment configuration file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="./configs/config_maisi_diff_model_train.json",
        help="Path to model training/inference configuration",
    )
    parser.add_argument(
        "--model_def", type=str, default="./configs/config_maisi.json", help="Path to model definition file"
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use for distributed training")
    parser.add_argument(
        "--metadata_path", 
        type=str, 
        default=None, 
        help="Path to metadata CSV file containing RescaleSlope and RescaleIntercept"
    )

    args = parser.parse_args()
    diff_model_create_training_data(args.env_config, args.model_config, args.model_def, args.num_gpus, args.metadata_path)

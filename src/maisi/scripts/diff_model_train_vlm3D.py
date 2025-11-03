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
from datetime import datetime
from pathlib import Path

import monai
import torch
import torch.distributed as dist
from monai.data import DataLoader, partition_dataset
from monai.networks.schedulers import RFlowScheduler
from monai.networks.schedulers.ddpm import DDPMPredictionType
from monai.transforms import Compose
from monai.utils import first
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .utils import define_instance


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
    filenames_train = json_data["training"]
    return sorted([os.path.basename(_item["image"]).replace(".nii.gz", "_emb.nii.gz") for _item in filenames_train])


    

    


def prepare_data(
    train_files: list,
    device: torch.device,
    cache_rate: float,
    num_workers: int = 2,
    batch_size: int = 1,
    include_body_region: bool = False,
) -> DataLoader:
    """
    Prepare training data.

    Args:
        train_files (list): List of training files.
        device (torch.device): Device to use for training.
        cache_rate (float): Cache rate for dataset.
        num_workers (int): Number of workers for data loading.
        batch_size (int): Mini-batch size.
        include_body_region (bool): Whether to include body region in data

    Returns:
        DataLoader: Data loader for training.
    """

    def _load_data_from_file(file_path, key):
        with open(file_path) as f:
            return torch.FloatTensor(json.load(f)[key])

    train_transforms_list = [
        monai.transforms.LoadImaged(keys=["image"]),
        monai.transforms.EnsureChannelFirstd(keys=["image"]),
        monai.transforms.Lambdad(keys="spacing", func=lambda x: _load_data_from_file(x, "spacing")),
        monai.transforms.Lambdad(keys="spacing", func=lambda x: x * 1e2),
        monai.transforms.Lambdad(keys="context_f", func=lambda x: _load_data_from_file(x, "findings_embeddings")),   # expects a (L, D) list or tensor,      
        monai.transforms.Lambdad(keys="context_i", func=lambda x: _load_data_from_file(x, "impression_embeddings"))  
    ]
    if include_body_region:
        train_transforms_list += [
            monai.transforms.Lambdad(
                keys="top_region_index", func=lambda x: _load_data_from_file(x, "top_region_index")
            ),
            monai.transforms.Lambdad(
                keys="bottom_region_index", func=lambda x: _load_data_from_file(x, "bottom_region_index")
            ),
            monai.transforms.Lambdad(keys="top_region_index", func=lambda x: x * 1e2),
            monai.transforms.Lambdad(keys="bottom_region_index", func=lambda x: x * 1e2),
        ]
    train_transforms = Compose(train_transforms_list)

    train_ds = monai.data.CacheDataset(
        data=train_files, transform=train_transforms, cache_rate=cache_rate, num_workers=num_workers
    )

    return DataLoader(train_ds, num_workers=6, batch_size=batch_size, shuffle=True)


def load_unet(args: argparse.Namespace, device: torch.device, logger: logging.Logger) -> torch.nn.Module:
    """
    Load the UNet model.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to load the model on.
        logger (logging.Logger): Logger for logging information.

    Returns:
        torch.nn.Module: Loaded UNet model.
    """
    unet = define_instance(args, "diffusion_unet_def").to(device)
    unet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(unet)
    start_epoch = 0
    best_loss = float("inf")
    current_lr = None
    if dist.is_initialized():
        unet = DistributedDataParallel(unet, device_ids=[device], find_unused_parameters=True)

    if args.existing_ckpt_filepath is None:
        logger.info("Training from scratch.")
        
    else:
        checkpoint_unet = torch.load(f"{args.existing_ckpt_filepath}", map_location=device, weights_only=False)
        if dist.is_initialized():
            unet.module.load_state_dict(checkpoint_unet["unet_state_dict"], strict=True)
        else:
            unet.load_state_dict(checkpoint_unet["unet_state_dict"], strict=True)
        logger.info(f"Pretrained checkpoint {args.existing_ckpt_filepath} loaded.")
        start_epoch = checkpoint_unet["epoch"]
        current_lr = checkpoint_unet["current_lr"] if "current_lr" in checkpoint_unet else None
        best_loss = checkpoint_unet["loss"] if "loss" in checkpoint_unet else float("inf")

    return unet, start_epoch, best_loss, current_lr


def calculate_scale_factor(train_loader: DataLoader, device: torch.device, logger: logging.Logger) -> torch.Tensor:
    """
    Calculate the scaling factor for the dataset.

    Args:
        train_loader (DataLoader): Data loader for training.
        device (torch.device): Device to use for calculation.
        logger (logging.Logger): Logger for logging information.

    Returns:
        torch.Tensor: Calculated scaling factor.
    """
    check_data = first(train_loader)
    z = check_data["image"].to(device)
    scale_factor = 1 / torch.std(z)
    logger.info(f"Scaling factor set to {scale_factor}.")

    if dist.is_initialized():
        dist.barrier()
        dist.all_reduce(scale_factor, op=torch.distributed.ReduceOp.AVG)
    logger.info(f"scale_factor -> {scale_factor}.")
    return scale_factor


def create_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    """
    Create optimizer for training.

    Args:
        model (torch.nn.Module): Model to optimize.
        lr (float): Learning rate.

    Returns:
        torch.optim.Optimizer: Created optimizer.
    """
    return torch.optim.Adam(params=model.parameters(), lr=lr)


def create_lr_scheduler(optimizer: torch.optim.Optimizer, total_steps: int) -> torch.optim.lr_scheduler.PolynomialLR:
    """
    Create learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to schedule.
        total_steps (int): Total number of training steps.

    Returns:
        torch.optim.lr_scheduler.PolynomialLR: Created learning rate scheduler.
    """
    return torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)


def train_one_epoch(
    epoch: int,
    unet: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.PolynomialLR,
    loss_pt: torch.nn.L1Loss,
    scaler: GradScaler,
    scale_factor: torch.Tensor,
    noise_scheduler: torch.nn.Module,
    num_images_per_batch: int,
    num_train_timesteps: int,
    device: torch.device,
    logger: logging.Logger,
    local_rank: int,
    amp: bool = True,
) -> torch.Tensor:
    """
    Train the model for one epoch.

    Args:
        epoch (int): Current epoch number.
        unet (torch.nn.Module): UNet model.
        train_loader (DataLoader): Data loader for training.
        optimizer (torch.optim.Optimizer): Optimizer.
        lr_scheduler (torch.optim.lr_scheduler.PolynomialLR): Learning rate scheduler.
        loss_pt (torch.nn.L1Loss): Loss function.
        scaler (GradScaler): Gradient scaler for mixed precision training.
        scale_factor (torch.Tensor): Scaling factor.
        noise_scheduler (torch.nn.Module): Noise scheduler.
        num_images_per_batch (int): Number of images per batch.
        num_train_timesteps (int): Number of training timesteps.
        device (torch.device): Device to use for training.
        logger (logging.Logger): Logger for logging information.
        local_rank (int): Local rank for distributed training.
        amp (bool): Use automatic mixed precision training.

    Returns:
        torch.Tensor: Training loss for the epoch.
    """
    if dist.is_initialized():
        include_body_region = unet.module.include_top_region_index_input
        include_modality = unet.module.num_class_embeds is not None
    else:
        include_body_region = unet.include_top_region_index_input
        include_modality = unet.num_class_embeds is not None

    if local_rank == 0:
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {epoch + 1}, lr {current_lr}.")

    _iter = 0
    loss_torch = torch.zeros(2, dtype=torch.float, device=device)

    unet.train()
    for train_data in train_loader:
        current_lr = optimizer.param_groups[0]["lr"]

        _iter += 1
        images = train_data["image"].to(device)
        images = images * scale_factor

        if include_body_region:
            top_region_index_tensor = train_data["top_region_index"].to(device)
            bottom_region_index_tensor = train_data["bottom_region_index"].to(device)
        # We trained with only CT in this version
        if include_modality:
            modality_tensor = torch.ones((len(images),), dtype=torch.long).to(device)
        spacing_tensor = train_data["spacing"].to(device)


        ### Context embeddings
        context_f = train_data["context_f"].to(device)     # (B, L, D) or (B, 1, D)
        context_i = train_data["context_i"].to(device)     # (B, L, D) or (B, 1, D)
        

        # text = train_data["Impressions_EN_embeddings"].to(device)  # (B, L, D) or (B, 1, D)
        if context_f.dim() == 2:                         # promote single token to sequence dim
            context_f = context_f.unsqueeze(1)
        if context_i.dim() == 2:                         # promote single token to sequence dim
            context_i = context_i.unsqueeze(1)
        
        ### concatenate context embeddings
        context = torch.cat((context_f, context_i), dim=1).to(device)  # Conc
        # print(f"context shape: {context.shape}")  # (B, 2, 512)
        # print(f"size of images: {images.shape}")  # (B, C, H, W)
        # context = torch.nn.functional.adaptive_avg_pool1d(context, 512)
        
    
        # print(f"context shape: {context.shape}")  # (B, L, D)

        # 🔁 CFG training trick: randomly drop conditioning
        drop_prob = 0.1  # 10% probability of dropping the condition
        if torch.rand(1).item() < drop_prob:
            context = torch.zeros_like(context)  # or context = None, if your model can handle it
        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=amp):
            noise = torch.randn_like(images)

            if isinstance(noise_scheduler, RFlowScheduler):
                timesteps = noise_scheduler.sample_timesteps(images)
            else:
                timesteps = torch.randint(0, num_train_timesteps, (images.shape[0],), device=images.device).long()

            noisy_latent = noise_scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)

            # Create a dictionary to store the inputs
            unet_inputs = {
                "x": noisy_latent,
                "timesteps": timesteps,
                "spacing_tensor": spacing_tensor,
                "context": context,            # <-- key name must be “context” !
            }
            # Add extra arguments if include_body_region is True
            if include_body_region:
                unet_inputs.update(
                    {
                        "top_region_index_tensor": top_region_index_tensor,
                        "bottom_region_index_tensor": bottom_region_index_tensor,
                    }
                )
            if include_modality:
                unet_inputs.update(
                    {
                        "class_labels": modality_tensor,
                    }
                )
            model_output = unet(**unet_inputs)

            if noise_scheduler.prediction_type == DDPMPredictionType.EPSILON:
                # predict noise
                model_gt = noise
            elif noise_scheduler.prediction_type == DDPMPredictionType.SAMPLE:
                # predict sample
                model_gt = images
            elif noise_scheduler.prediction_type == DDPMPredictionType.V_PREDICTION:
                # predict velocity
                model_gt = images - noise
            else:
                raise ValueError(
                    "noise scheduler prediction type has to be chosen from ",
                    f"[{DDPMPredictionType.EPSILON},{DDPMPredictionType.SAMPLE},{DDPMPredictionType.V_PREDICTION}]",
                )

            loss = loss_pt(model_output.float(), model_gt.float())

        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        loss_torch[0] += loss.item()
        loss_torch[1] += 1.0

        if local_rank == 0 and _iter % 1000 == 0:
            logger.info(
                "[{0}] epoch {1}, iter {2}/{3}, loss: {4:.4f}, lr: {5:.12f}.".format(
                    str(datetime.now())[:19], epoch + 1, _iter, len(train_loader), loss.item(), current_lr
                )
            )

    if dist.is_initialized():
        dist.all_reduce(loss_torch, op=torch.distributed.ReduceOp.SUM)

    return loss_torch, current_lr


def save_checkpoint(
    epoch: int,
    unet: torch.nn.Module,
    loss_torch_epoch: float,
    num_train_timesteps: int,
    scale_factor: torch.Tensor,
    ckpt_folder: str,
    args: argparse.Namespace,
    current_lr: float = None,
    out_model_filename: str = None,
) -> None:
    """
    Save checkpoint.

    Args:
        epoch (int): Current epoch number.
        unet (torch.nn.Module): UNet model.
        loss_torch_epoch (float): Training loss for the epoch.
        num_train_timesteps (int): Number of training timesteps.
        scale_factor (torch.Tensor): Scaling factor.
        ckpt_folder (str): Checkpoint folder path.
        args (argparse.Namespace): Configuration arguments.
    """
    unet_state_dict = unet.module.state_dict() if dist.is_initialized() else unet.state_dict()
    torch.save(
        {
            "epoch": epoch + 1,
            "loss": loss_torch_epoch,
            "num_train_timesteps": num_train_timesteps,
            "scale_factor": scale_factor,
            "unet_state_dict": unet_state_dict,
            "args": args,
            "current_lr": current_lr if current_lr is not None else None
        },
        f"{ckpt_folder}/{out_model_filename}",
    )


def diff_model_train(
    env_config_path: str, model_config_path: str, model_def_path: str, num_gpus: int, amp: bool = True, structured_report: bool = False
) -> None:
    """
    Main function to train a diffusion model.

    Args:
        env_config_path (str): Path to the environment configuration file.
        model_config_path (str): Path to the model configuration file.
        model_def_path (str): Path to the model definition file.
        num_gpus (int): Number of GPUs to use for training.
        amp (bool): Use automatic mixed precision training.
    """
    args = load_config(env_config_path, model_config_path, model_def_path)
    local_rank, world_size, device = initialize_distributed(num_gpus)
    logger = setup_logging("training")

    logger.info(f"Using {device} of {world_size}")

    if local_rank == 0:
        logger.info(f"[config] ckpt_folder -> {args.model_dir}.")
        logger.info(f"[config] data_root -> {args.embedding_base_dir}.")
        logger.info(f"[config] data_list -> {args.json_data_list}.")
        logger.info(f"[config] lr -> {args.diffusion_unet_train['lr']}.")
        logger.info(f"[config] num_epochs -> {args.diffusion_unet_train['n_epochs']}.")
        logger.info(f"[config] num_train_timesteps -> {args.noise_scheduler['num_train_timesteps']}.")

        Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    unet, start_epoch, best_loss, current_lr = load_unet(args, device, logger)
    if current_lr is not None:
        args.diffusion_unet_train["lr"] = current_lr
        logger.info(f"loaded lr -> {args.diffusion_unet_train['lr']}.")
    print(f"number of parameters in UNet: {sum(p.numel() for p in unet.parameters() if p.requires_grad)}")
    # print(unet)
    noise_scheduler = define_instance(args, "noise_scheduler")
    logger.info(f"best loss: {best_loss:.4f}.")

        # Fix at line 435 - add distributed check
    if dist.is_initialized():
        include_body_region = unet.module.include_top_region_index_input
        include_modality = unet.module.num_class_embeds is not None
    else:
        include_body_region = unet.include_top_region_index_input
        include_modality = unet.num_class_embeds is not None

    filenames_train = load_filenames(args.json_data_list)
    if local_rank == 0:
        logger.info(f"num_files_train: {len(filenames_train)}")

    train_files = []
    for _i in range(len(filenames_train)):
        str_img = os.path.join(args.embedding_base_dir, filenames_train[_i])
        if not os.path.exists(str_img):
            continue

        if structured_report:
            str_info = os.path.join(args.embedding_base_dir, filenames_train[_i]) + "structured_separated.json"
        else: 
            str_info = os.path.join(args.embedding_base_dir, filenames_train[_i]) + ".json"
        train_files_i = {"image": str_img, "spacing": str_info, "context_i": str_info, "context_f": str_info}
        if include_body_region:
            train_files_i["top_region_index"] = str_info
            train_files_i["bottom_region_index"] = str_info
        train_files.append(train_files_i)
    if dist.is_initialized():
        train_files = partition_dataset(
            data=train_files, shuffle=True, num_partitions=dist.get_world_size(), even_divisible=True
        )[local_rank]

    train_loader = prepare_data(
        train_files,
        device,
        args.diffusion_unet_train["cache_rate"],
        batch_size=args.diffusion_unet_train["batch_size"],
        include_body_region=include_body_region,
    )

    scale_factor = calculate_scale_factor(train_loader, device, logger)
    optimizer = create_optimizer(unet, args.diffusion_unet_train["lr"])

    total_steps = (args.diffusion_unet_train["n_epochs"] * len(train_loader.dataset)) / args.diffusion_unet_train[
        "batch_size"
    ]
    lr_scheduler = create_lr_scheduler(optimizer, total_steps)
    loss_pt = torch.nn.L1Loss()
    scaler = GradScaler("cuda")

    torch.set_float32_matmul_precision("highest")
    # torch.set_float32_matmul_precision("high")
    logger.info("torch.set_float32_matmul_precision -> highest.")
    print(f"start_epoch: {start_epoch}, total epochs: {args.diffusion_unet_train['n_epochs']}")

    for epoch in tqdm(range(start_epoch, args.diffusion_unet_train["n_epochs"])):
        loss_torch, current_lr = train_one_epoch(
            epoch,
            unet,
            train_loader,
            optimizer,
            lr_scheduler,
            loss_pt,
            scaler,
            scale_factor,
            noise_scheduler,
            args.diffusion_unet_train["batch_size"],
            args.noise_scheduler["num_train_timesteps"],
            device,
            logger,
            local_rank,
            amp=amp,
        )

        loss_torch = loss_torch.tolist()
        if torch.cuda.device_count() == 1 or local_rank == 0:
            loss_torch_epoch = loss_torch[0] / loss_torch[1]
            logger.info(f"epoch {epoch + 1} average loss: {loss_torch_epoch:.4f}.")


            if loss_torch_epoch < best_loss:
                best_loss = loss_torch_epoch
                logger.info(f"Saving model at epoch {epoch + 1} with loss {best_loss:.4f}.")
                out_model_filename = args.model_filename.replace(".pt", f"_best.pt")
                save_checkpoint(
                    epoch,
                    unet,
                    best_loss,
                    args.noise_scheduler["num_train_timesteps"],
                    scale_factor,
                    args.model_dir,
                    args,
                    current_lr= current_lr,
                    out_model_filename=out_model_filename,
                )
            else:
                logger.info(f"Model at epoch {epoch + 1} did not improve, current best loss: {best_loss:.4f}.")
                
                out_model_filename = args.model_filename.replace(".pt", f"_last.pt")
                save_checkpoint(
                    epoch,
                    unet,
                    best_loss,
                    args.noise_scheduler["num_train_timesteps"],
                    scale_factor,
                    args.model_dir,
                    args,
                    current_lr= current_lr,
                    out_model_filename=out_model_filename,
                )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Training")
    parser.add_argument(
        "--env_config",
        type=str,
        default="./configs/environment_maisi_diff_model.json",
        help="Path to environment configuration file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="./configs/config_maisi_diff_model.json",
        help="Path to model training/inference configuration",
    )
    parser.add_argument(
        "--model_def", type=str, default="./configs/config_maisi.json", help="Path to model definition file"
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use for training")
    parser.add_argument("--no_amp", dest="amp", action="store_false", help="Disable automatic mixed precision training")
    parser.add_argument("--structured_report", dest="structured_report", action="store_true", help="Enable structured report training")

    args = parser.parse_args()
    print(f"Arguments: {args}")
    diff_model_train(args.env_config, args.model_config, args.model_def, args.num_gpus, args.amp, args.structured_report)

"""
Report2CT inference: synthesize a 3D chest CT volume from a free-text radiology report.

Example:
    python inference.py \
        --findings "Peripheral and central consolidations and areas of ground glass are observed in both lungs." \
        --impression "Findings consistent with viral pneumonia in both lungs." \
        --output ./outputs/report2ct_sample.nii.gz
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import nibabel as nib
import numpy as np
import torch
from monai.inferers.inferer import SlidingWindowInferer
from monai.utils import set_determinism
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.maisi.scripts.diff_model_setting import setup_logging
from src.maisi.scripts.utils import define_instance, dynamic_infer

MODEL_DEF = "./vlm3D_work_dir/config_maisi_2560.json"
UNET_CKPT = "./vlm3D_work_dir/models/diff_unet_ckpt_FI_2560_multi_best_inference.pt"
AUTOENCODER_CKPT = "./vlm3D_work_dir/models/autoencoder_epoch273.pt"

# Mean-pooled embeddings of these encoders are concatenated: 1024 + 768 + 768 = 2560 dims.
TEXT_ENCODERS = [
    "abhinand/MedEmbed-large-v0.1",
    "medicalai/ClinicalBERT",
    "microsoft/BiomedVLP-CXR-BERT-specialized",
]
MAX_SEQ_LEN = 512

# Report2CT was trained on volumes resampled to this size (latent size 120 x 120 x 64).
OUTPUT_SIZE = (480, 480, 256)


class ReportEncoder:
    """Encodes report text with the three medical text encoders used during training."""

    def __init__(self, device: torch.device):
        self.device = device
        self.tokenizers = [AutoTokenizer.from_pretrained(name, trust_remote_code=True) for name in TEXT_ENCODERS]
        self.models = [
            AutoModel.from_pretrained(name, trust_remote_code=True).eval().to(device) for name in TEXT_ENCODERS
        ]

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        """
        Embed one text with every encoder and concatenate the mean-pooled outputs.

        Args:
            text (str): Report section (findings or impression).

        Returns:
            torch.Tensor: Text embedding of shape (2560,).
        """
        pooled = []
        for tokenizer, model in zip(self.tokenizers, self.models):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            hidden = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled.append((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9))
        return torch.cat(pooled, dim=-1).squeeze(0).float().cpu()

    def encode_report(self, findings: str, impression: str) -> torch.Tensor:
        """
        Build the UNet cross-attention context from the two report sections.

        Returns:
            torch.Tensor: Context of shape (1, 2, 2560), ordered [findings, impression].
        """
        return torch.stack([self.encode(findings), self.encode(impression)]).unsqueeze(0)


def load_report2ct(
    device: torch.device,
    unet_ckpt: str = UNET_CKPT,
    autoencoder_ckpt: str = AUTOENCODER_CKPT,
    model_def: str = MODEL_DEF,
) -> tuple:
    """
    Build the MAISI autoencoder and the report-conditional diffusion UNet and load their weights.

    Returns:
        tuple: Model definition, autoencoder, UNet, and latent scale factor.
    """
    with open(model_def, "r") as f:
        model_args = argparse.Namespace(**json.load(f))

    autoencoder = define_instance(model_args, "autoencoder_def").to(device)
    autoencoder.load_state_dict(torch.load(autoencoder_ckpt, map_location=device, weights_only=True))

    unet = define_instance(model_args, "diffusion_unet_def").to(device)
    checkpoint = torch.load(unet_ckpt, map_location=device, weights_only=True)
    unet.load_state_dict(checkpoint["unet_state_dict"], strict=True)
    scale_factor = torch.as_tensor(checkpoint["scale_factor"], device=device).float()

    return model_args, autoencoder.eval(), unet.eval(), scale_factor


@torch.no_grad()
def generate_ct(
    model_args: argparse.Namespace,
    autoencoder: torch.nn.Module,
    unet: torch.nn.Module,
    scale_factor: torch.Tensor,
    context: torch.Tensor,
    spacing: tuple,
    num_inference_steps: int = 500,
    cfg_scale: float | None = None,
    seed: int | None = 0,
) -> np.ndarray:
    """
    Sample a latent with the rectified-flow scheduler and decode it to a CT volume.

    Args:
        model_args (argparse.Namespace): Model definition loaded by `load_report2ct`.
        autoencoder (torch.nn.Module): MAISI autoencoder used for decoding.
        unet (torch.nn.Module): Report-conditional diffusion UNet.
        scale_factor (torch.Tensor): Latent scale factor stored in the UNet checkpoint.
        context (torch.Tensor): Report embedding from `ReportEncoder.encode_report`, shape (1, 2, 2560).
        spacing (tuple): Voxel spacing in mm of the generated volume.
        num_inference_steps (int): Number of rectified-flow sampling steps.
        cfg_scale (float | None): Classifier-free guidance scale; None disables guidance.
        seed (int | None): Random seed for the initial noise.

    Returns:
        np.ndarray: CT volume in Hounsfield units, int16, shape `OUTPUT_SIZE`.
    """
    device = scale_factor.device
    if seed is not None:
        set_determinism(seed)

    divisor = 2 ** (len(model_args.diffusion_unet_def["num_channels"]) - 2)
    latent_shape = (1, model_args.latent_channels, *[s // divisor for s in OUTPUT_SIZE])
    image = torch.randn(latent_shape, dtype=torch.float16, device=device)

    noise_scheduler = define_instance(model_args, "noise_scheduler")
    noise_scheduler.set_timesteps(
        num_inference_steps=num_inference_steps,
        input_img_size_numel=torch.prod(torch.tensor(latent_shape[2:])),
    )

    context = context.to(device)
    spacing_tensor = (torch.tensor(spacing) * 1e2).half().unsqueeze(0).to(device)
    modality_tensor = torch.ones((1,), dtype=torch.long, device=device)  # class label 1 = CT, as in training
    if cfg_scale is not None:
        # Batch the conditional and unconditional (all-zero context) passes; training dropped the context 15% of the time.
        context = torch.cat([context, torch.zeros_like(context)])
        spacing_tensor = spacing_tensor.repeat(2, 1)
        modality_tensor = modality_tensor.repeat(2)

    all_timesteps = noise_scheduler.timesteps
    all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype)))
    with torch.amp.autocast("cuda"):
        for t, next_t in tqdm(zip(all_timesteps, all_next_timesteps), total=len(all_timesteps)):
            x = image if cfg_scale is None else torch.cat([image, image])
            model_output = unet(
                x=x,
                timesteps=t.repeat(len(x)).to(device),
                spacing_tensor=spacing_tensor,
                context=context,
                class_labels=modality_tensor,
            )
            if cfg_scale is not None:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = output_uncond + cfg_scale * (output_cond - output_uncond)
            image, _ = noise_scheduler.step(model_output, t, image, next_t)

        inferer = SlidingWindowInferer(
            roi_size=[80, 80, 80],
            sw_batch_size=1,
            progress=True,
            mode="gaussian",
            overlap=0.4,
            sw_device=device,
            device=device,
        )
        decoded = dynamic_infer(inferer, lambda z: autoencoder.decode_stage_2_outputs(z / scale_factor), image.float())

    # The autoencoder output in [0, 1] maps linearly to [-1000, 1000] HU.
    volume = decoded.squeeze().float().cpu().numpy() * 2000 - 1000
    return np.clip(volume, -1000, 1000).astype(np.int16)


def save_ct(volume: np.ndarray, spacing: tuple, output_path: str) -> None:
    """Save a generated volume as NIfTI with the given voxel spacing."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    nib.save(nib.Nifti1Image(volume, affine=np.diag([*spacing, 1.0])), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a 3D chest CT from a radiology report with Report2CT.")
    parser.add_argument("--findings", type=str, required=True, help="Findings section of the report")
    parser.add_argument("--impression", type=str, required=True, help="Impression section of the report")
    parser.add_argument("--output", type=str, default="./outputs/report2ct.nii.gz", help="Output NIfTI path")
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=[0.87, 0.87, 1.35],
        help="Voxel spacing (mm) of the 480x480x256 output, i.e. field of view / output size",
    )
    parser.add_argument("--num_steps", type=int, default=500, help="Number of sampling steps")
    parser.add_argument("--cfg_scale", type=float, default=None, help="Classifier-free guidance scale (off by default)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--unet_ckpt", type=str, default=UNET_CKPT, help="Diffusion UNet checkpoint")
    parser.add_argument("--autoencoder_ckpt", type=str, default=AUTOENCODER_CKPT, help="MAISI autoencoder checkpoint")
    args = parser.parse_args()

    logger = setup_logging("report2ct")
    device = torch.device("cuda")

    logger.info("Encoding report...")
    encoder = ReportEncoder(device)
    context = encoder.encode_report(args.findings, args.impression)
    del encoder
    torch.cuda.empty_cache()

    logger.info("Loading Report2CT...")
    model_args, autoencoder, unet, scale_factor = load_report2ct(device, args.unet_ckpt, args.autoencoder_ckpt)

    logger.info(f"Generating CT: size {OUTPUT_SIZE}, spacing {args.spacing}, seed {args.seed}")
    volume = generate_ct(
        model_args,
        autoencoder,
        unet,
        scale_factor,
        context,
        args.spacing,
        num_inference_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
    )
    save_ct(volume, args.spacing, args.output)
    logger.info(f"Saved {args.output}")


if __name__ == "__main__":
    main()

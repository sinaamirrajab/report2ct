# Report2CT: Radiology Report-Conditional 3D CT Generation

[![License](https://img.shields.io/badge/License-Academic%20Research%20License-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2509.14780-b31b1b.svg)](https://arxiv.org/abs/2509.14780)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.5.0-green.svg)](https://monai.io/)

**Authors:** Sina Amirrajab¹, Zohaib Salahuddin¹, Sheng Kuang¹, Henry C. Woodruff¹˒², Philippe Lambin¹˒²

¹ The D-Lab, Department of Precision Medicine, GROW - Research Institute for Oncology and Reproduction, Maastricht University, 6220 MD Maastricht, The Netherlands

² Department of Radiology and Nuclear Medicine, GROW - Research Institute for Oncology and Reproduction, Maastricht University, Medical Center+, 6229 HX Maastricht, The Netherlands

![Report2CT Overview](paper/visuals/Untitled%20design.gif)

> 🏆 **Winner of MICCAI 2025 VLM3D Challenge - Task 4: Text-Conditional CT Generation >** [Leaderboard](https://ctgen.vlm3dchallenge.com/evaluation/test/leaderboard/)

## Overview

Report2CT is a radiology report-conditional latent diffusion framework for synthesizing 3D chest CT volumes directly from free-text radiology reports. This repository contains the implementation and research materials for our work that **ranked 1st in Task 4 of the VLM3D Challenge at MICCAI 2025** on Text-Conditional CT Generation.

📄 **Paper**: [arXiv:2509.14780](https://arxiv.org/abs/2509.14780)

## Qualitative Results

![Qualitative Comparison](paper/visuals/comparison.jpg)

**Figure:** Qualitative comparison between our proposed Report2CT model and the GenerateCT model for different abnormalities (lung pneumonia in orange, pleural effusion in green, and cardiomegaly in blue). The red zoomed-in region highlights the consistency of Report2CT across orthogonal views. Both findings and impressions were used for synthesis; for brevity, only impression texts are shown.

## Installation

```bash
# Clone the repository
git clone https://github.com/sinaamirrajab/report2ct.git
cd report2ct

# Create an environment and install dependencies
conda create -n report2ct python=3.10 -y
conda activate report2ct
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.6.0 (CUDA 12.4) and MONAI 1.5.0 on an NVIDIA H100.

## Pretrained Weights

| File | Contents | Size |
|------|----------|------|
| [report2ct_weights.zip](https://drive.google.com/file/d/1JQuGAOQpHrh0lccLATxut9OC5nAi7V11/view?usp=sharing) | Report2CT diffusion UNet + MAISI VAE | 943 MB |

Download and extract from the repository root:

```bash
gdown --fuzzy "https://drive.google.com/file/d/1JQuGAOQpHrh0lccLATxut9OC5nAi7V11/view?usp=sharing" -O report2ct_weights.zip
unzip report2ct_weights.zip
```

This creates:

```
vlm3D_work_dir/models/
├── autoencoder_epoch273.pt                          # MAISI VAE (NVIDIA)
└── diff_unet_ckpt_FI_2560_multi_best_inference.pt   # Report2CT diffusion UNet
```

The three text encoders ([MedEmbed-large-v0.1](https://huggingface.co/abhinand/MedEmbed-large-v0.1), [ClinicalBERT](https://huggingface.co/medicalai/ClinicalBERT), [BiomedVLP-CXR-BERT-specialized](https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized)) are downloaded automatically from Hugging Face on first use.

## Inference

Inference needs a CUDA GPU with ~40 GB of memory. Generating one 480 × 480 × 256 CT with 500 sampling steps takes about 1 minute on an H100.

### Notebook

Open [vlm3d_inference.ipynb](vlm3d_inference.ipynb) from the repository root and run all cells. The notebook downloads the weights if they are missing, encodes the report, generates the CT, saves it to `outputs/report2ct_sample.nii.gz` and shows three orthogonal slices. To use your own report, edit `findings`, `impression` and `spacing` in step 2.

### Command line

```bash
python inference.py \
    --findings "Peripheral and central consolidations and areas of ground glass are observed in both lungs." \
    --impression "Findings consistent with viral pneumonia in both lungs." \
    --output outputs/report2ct_sample.nii.gz
```

| Option | Default | Description |
|--------|---------|-------------|
| `--spacing` | `0.87 0.87 1.35` | Voxel spacing (mm) of the output: original spacing × original size / (480, 480, 256) |
| `--num_steps` | `500` | Number of sampling steps |
| `--cfg_scale` | off | Classifier-free guidance scale |
| `--seed` | `0` | Random seed |

The output is a NIfTI volume of 480 × 480 × 256 voxels in Hounsfield units, clipped to [-1000, 1000].

## Training

### 1. Data Preparation

Prepare your CT data and radiology reports in the expected format. The model expects:

- **CT Volumes**: 3D chest CT scans in NIfTI format
- **Radiology Reports**: Free-text reports containing both Findings and Impression sections
- **Metadata**: Voxel spacing information for proper conditioning

### 2. Image Embedding Generation

First, generate image embeddings using the pretrained MAISI VAE-GAN model:

```bash
python -m src.maisi.scripts.vlm3d_image_embedding
```

### 3. Training

Train the report-conditional diffusion model:

```bash
torchrun --nproc_per_node=2 -m src.maisi.scripts.diff_model_train_vlm3D_2560_multi_text \
    --env_config ./vlm3D_work_dir/environment_maisi_diff_model_vlm3D_FI_2560_multi.json \
    --model_config ./vlm3D_work_dir/config_maisi_diff_model_vlm3D.json \
    --model_def ./vlm3D_work_dir/config_maisi_2560.json \
    --num_gpus 2
```

## License

This project is licensed under the **Academic Research License – NonCommercial, No Derivatives**. See the [LICENSE](LICENSE) file for details.


## Citation

If you use Report2CT in your research, please cite our work:

```bibtex
@article{amirrajab2025report2ct,
    title={Report2CT: Radiology Report-Conditional 3D CT Generation with Multi-Encoder Latent-Diffusion Model},
    author={Amirrajab, Sina and Salahuddin, Zohaib and Kuang, Sheng and Woodruff, Henry C. and Lambin, Philippe},
    journal={Medical Image Analysis},
    year={2025},
    note={Winner of MICCAI 2025 VLM3D Challenge - Task 4}
}
```

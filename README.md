# Radiology Report-Conditional 3D CT Generation with Multi-Encoder Latent-Diffusion Model (Report2CT)

**Authors:** Sina Amirrajab¹, Zohaib Salahuddin¹, Sheng Kuang¹, Henry C. Woodruff¹˒², Philippe Lambin¹˒²

¹ The D-Lab, Department of Precision Medicine, GROW - Research Institute for Oncology and Reproduction, Maastricht University, 6220 MD Maastricht, The Netherlands

² Department of Radiology and Nuclear Medicine, GROW - Research Institute for Oncology and Reproduction, Maastricht University, Medical Center+, 6229 HX Maastricht, The Netherlands

![Report2CT Overview](paper/visuals/Untitled%20design.gif)

## Overview

Report2CT is a radiology report-conditional latent diffusion framework for synthesizing 3D chest CT volumes directly from free-text radiology reports. This repository contains the implementation and research materials for our work that **ranked 1st in Task 4 of the VLM3D Challenge at MICCAI 2025** on Text-Conditional CT Generation.

## Abstract

**Background:** Text-to-image diffusion models have recently advanced medical image synthesis, but applications to 3D CT remain limited. Existing approaches rely on simplified prompts, neglecting the rich semantic detail in full radiology reports, which reduces text–image alignment and clinical fidelity.

**Purpose:** We propose Report2CT, a radiology report–conditional latent diffusion framework for synthesizing 3D chest CT volumes directly from free-text radiology reports, incorporating both Findings and Impression sections.

**Materials and Methods:** Report2CT integrates three pretrained medical text encoders (BiomedVLP-CXR-BERT, MedEmbed, and ClinicalBERT) to capture nuanced clinical context. Radiology reports and voxel-spacing information condition a 3D latent diffusion model trained on 20,000 CT volumes from the CT-RATE dataset. Model performance was evaluated using Fréchet Inception Distance (FID) for distributional similarity and CLIP-based metrics for semantic alignment, with additional qualitative comparisons against GenerateCT.

**Results:** Report2CT generated anatomically consistent CT volumes with superior visual quality and text–image alignment compared with GenerateCT. Multi-encoder conditioning improved CLIP scores, indicating stronger preservation of fine-grained clinical details. Classifier-free guidance further enhanced alignment with only a minor trade-off in FID. Report2CT ranked 1st in Task 4 of the VLM3D Challenge at MICCAI 2025 on Text-Conditional CT Generation and achieved state-of-the-art performance across all evaluation metrics.

**Conclusion:** By leveraging complete radiology reports and multi-encoder text conditioning, Report2CT advances state-of-the-art 3D CT synthesis, producing clinically faithful and high-quality synthetic data. These findings highlight the potential of report-driven generation for data augmentation, fairness, and robustness in medical imaging AI.

**Keywords:** Radiology reports; text-to-CT generation; latent diffusion model; synthetic medical imaging; data augmentation; multimodal AI

## Qualitative Results

![Qualitative Comparison](paper/visuals/comparison.jpg)

**Figure:** Qualitative comparison between our proposed Report2CT model and the GenerateCT model for different abnormalities (lung pneumonia in orange, pleural effusion in green, and cardiomegaly in blue). The red zoomed-in region highlights the consistency of Report2CT across orthogonal views. Both findings and impressions were used for synthesis; for brevity, only impression texts are shown.

## Status

🚧 **Code Coming Soon!** 🚧

The complete source code implementation will be released after the **MICCAI 2025 VLM3D Challenge** concludes. Stay tuned for updates!

For questions or collaboration opportunities, please open an issue or contact the authors.


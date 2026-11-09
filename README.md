# ARED-Net and CARE-MammoRestore

PyTorch implementation *ARED-Net and CARE-MammoRestore: An Anatomy-Aware Framework for Tattoo Artifact Suppression and Breast Tumor Analysis in Mammography* by Manal Almuzini and Muhammad Shoaib.

This software was developed for research purpose. The study reports a restricted Sunway Medical Centre cohort of 500 cases (350 training, 75 validation, 75 testing). The study data, exact partition, original source code, trained weights and per-case predictions were unavailable for this reconstruction. The paper's reported performance has not been reproduced. This code is intended for research evaluation.

## Setup and smoke test

Python 3.12 was used for the smoke test. Install a PyTorch build appropriate for your CPU or CUDA environment and the dependencies below.

```bash
python -m pip install -r requirements.txt
python smoke_test.py
```

The smoke test uses temporary artificial arrays to exercise all three training stages, checkpoint reload and paired evaluation. It does not assess model performance.

## Prepare authorized study inputs

Provide a private CSV using the header in `manifest_template.csv`. Each row represents one analysis case or image with one binary diagnosis. Use de-identified patient and case keys. Keep all images for a patient in one partition. Use the study's original patient-grouped, class/severity-stratified split to compare against published numbers. The loader rejects patient leakage and duplicate case IDs. The severity column records the study grouping and is available for downstream subgroup analysis.

- `split` accepts `train`, `val` and `test`.
- `label` is 0 for benign and 1 for malignant.
- `image`, `tumor_mask` and `tattoo_mask` are paths relative to the manifest, or absolute paths.
- Inputs are aligned 2-D NumPy arrays. Prepare diagnostic grayscale polarity and crop acquisition borders before export. The loader resizes and min–max normalizes the image. Masks use binary 0/1 values and nearest-neighbor resizing. Use an all-zero tattoo mask for a confirmed artifact-free image. Obtain the annotation before using images with unknown tattoo status.
- Training applies synchronized rotation, translation and scaling, plus image-only intensity perturbations. Horizontal flipping is omitted because the input format lacks laterality information.

## Train and evaluate

```bash
python run.py train --manifest private/manifest.csv --output runs/reference --size 256 256
python run.py evaluate --manifest private/manifest.csv --output runs/reference --size 256 256
```

The training command selects and freezes ARED-Net, then trains and freezes CARE-MammoRestore before training the tumor analyzer on paired raw and restored inputs. The analyzer is selected by the mean of validation Dice and AUROC. The independent test command uses one frozen analyzer for both input conditions. It saves `paired_test.csv` and `metrics.json` locally. Output includes tattoo and tumor overlap measures, raw and restored classification metrics, paired gains, patient-level bootstrap intervals for segmentation gains and outside-mask identity error. Segmentation metrics average cases within patient, then weight patients equally. Classification metrics are case-level. Empty/empty masks score 1 for Dice and IoU, and undefined sensitivity, precision and specificity use zero. The outside-mask error is zero by construction where the hard edit gate is zero. The code does not estimate identity against a manually annotated artifact region.

## Implementation choices and scope

- **ARED-Net** implements masked context reconstruction, a gated image/residual encoder, lesion and artifact feature separation, a binary Dirichlet evidence head and cosine-based anchor propagation. Anchor groups are represented by their within-image mean features.
- **CARE-MammoRestore** implements confidence-adaptive bounded editing, CNN and windowed Transformer features, cross-window context, one-level Haar bands, uncertainty-based mixture weights and a lesion-protection head. Training includes a within-batch residual-transfer stream, boundary perturbations and anatomical preservation losses. Residual-transfer targets treat annotated artifact-free recipient regions as clean. Early in training, some batches can have no eligible donor pixels and contribute zero paired reconstruction loss.
- **Tumor analyzer** implements lesion-weighted pooling and joint segmentation and diagnosis with class weights calculated on training cases. It uses the manuscript's AdamW settings, batch size, maximum epochs and early stopping by default.
- Network widths, spatial resolution, augmentation ranges, seed, numerical thresholds, loss weights, warm-up duration and ARED/CARE training schedules are reference choices. Training is full precision. Validation for CARE uses the reference composite loss with stochastic perturbations and residual transfer.
- The package covers the main training and paired evaluation path. Original baseline implementations, component ablations, severity-stratified reports, paired-reference SSIM/PSNR/LPIPS reporting and manuscript significance tests are outside this release. 



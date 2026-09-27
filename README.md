# Gal-FNO: training and basin inference

This repository contains two Python scripts:

- `train_galfno.py` trains a five-member Gal-FNO ensemble using the supplied 10 m and 20 m training tensors.
- `inference_galfno.py` applies those five checkpoints to the supplied 20 m basin tensors and saves chlorophyll-a mean and uncertainty maps.

The data are distributed separately through Zenodo. "https://doi.org/10.5281/zenodo.22996402"

## 1. Set up the local project folder

Download or clone this GitHub repository. Its local directory is the *project folder*. Download the files from `https://doi.org/10.5281/zenodo.22996402` and arrange them as shown below. Extract the inference tensor ZIP containing the 36 `.bin` files; they must be directly inside `GalFNO_Inference_Tensors_20m_21_23/`.

```text
gal-fno/                              # Local project folder; name can vary
├── README.md
├── train_galfno.py
├── inference_galfno.py
└── data/
    ├── inputs_10m.bin
    ├── outputs_10m.bin
    ├── inputs_20m.bin
    ├── outputs_20m.bin
    ├── Bathymetry_20m.npy
    ├── bathymetry/
    │   ├── beta_mask_10m.bin
    │   ├── beta_mask_20m.bin
    │   ├── bathymetry_10m.bin
    │   └── bathymetry_20m.bin
    └── GalFNO_Inference_Tensors_20m_21_23/
        ├── ..._10CH_YYYYMMDD.bin
        └── ...
```

Create `data/` and `data/bathymetry/` locally. The eight training files may be downloaded as separate files from Zenodo; their placement in the project folder is what matters to the scripts. `Bathymetry_20m.npy` and the tensor ZIP are needed for inference, not for training. The scripts create their own `results/` subfolders.

You need a Python environment with **PyTorch, NumPy, and Matplotlib**. Install a PyTorch build suitable for your computer using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/), then install NumPy and Matplotlib in the same environment:

```bash
python3 -m pip install numpy matplotlib
```

The scripts select CUDA, Apple MPS, or CPU when available. Training and full-basin inference may require substantial memory and time.

## 2. Train the five models

From the project folder, run:

```bash
python3 train_galfno.py
```

The script reads these **raw, headerless `float32`** files. Their shapes and the ordering of samples and channels must match the supplied dataset:

| File | Shape |
| --- | --- |
| `inputs_10m.bin` | `(4000, 40, 116, 10)` |
| `outputs_10m.bin` | `(4000, 40, 116)` |
| `inputs_20m.bin` | `(4000, 20, 58, 10)` |
| `outputs_20m.bin` | `(4000, 20, 58)` |
| `beta_mask_10m.bin`, `bathymetry_10m.bin` | `(40, 116)` each |
| `beta_mask_20m.bin`, `bathymetry_20m.bin` | `(20, 58)` each |

The beta masks are used in training. The bathymetry `.bin` files are used by the evaluation figures produced after training. The script uses the supplied normalization and bio-optical constants; these files are prepared model inputs, rather than raw satellite scenes.

Training runs five ensemble members for 50 epochs each. It writes:

```text
results/
├── checkpoints/
│   ├── Gal-FNO_Fourier_Ensemble_1.pth
│   ├── Gal-FNO_Fourier_Ensemble_2.pth
│   ├── Gal-FNO_Fourier_Ensemble_3.pth
│   ├── Gal-FNO_Fourier_Ensemble_4.pth
│   └── Gal-FNO_Fourier_Ensemble_5.pth
└── figures/
    ├── Gal-FNO_Fourier_Ensemble_Loss.png
    └── evaluation/                    # Compliance, spatial, and resolution figures
```

The five `.pth` files are required for the next step. A reader who already has these exact compatible checkpoints can place them in `results/checkpoints/` and proceed directly to inference.

## 3. Run inference

Keep `Bathymetry_20m.npy` in `data/`, and extract the Zenodo ZIP so the inference `.bin` files are directly inside `data/GalFNO_Inference_Tensors_20m_21_23/`. Then run:

```bash
python3 inference_galfno.py
```

The script finds files matching `*_10CH_*.bin`; the last underscore-separated part of each filename must be a date in `YYYYMMDD` format. Each file is a raw `float32` tensor with ten channels. Its height must match the 2-D bathymetry array, and its width must not exceed the bathymetry width. The scripts use the same 20 m chlorophyll normalization values (`mean = 0.7769`, `std = 0.3453`), so `Gal-FNO_Ratio_Diagnostics.txt` is not an additional input.

If the extracted tensor folder is stored elsewhere, pass its path explicitly:

```bash
python3 inference_galfno.py --tensors-dir /path/to/GalFNO_Inference_Tensors_20m_21_23
```

Both scripts also accept `--base-dir /path/to/gal-fno` when the project folder is elsewhere. If omitted, `--base-dir` defaults to the directory containing the script.

## 4. Plot the saved inference results

For each input date `YYYYMMDD`, inference writes four 2-D NumPy arrays to `results/inference/`:

| Output file | Content |
| --- | --- |
| `GalFNO_Mean_YYYYMMDD.npy` | Chlorophyll-a mean prediction |
| `GalFNO_AleatorySD_YYYYMMDD.npy` | Aleatory standard deviation |
| `GalFNO_EpistemicSD_YYYYMMDD.npy` | Ensemble-based epistemic standard deviation |
| `GalFNO_StdTotal_YYYYMMDD.npy` | Total standard deviation, computed as the square root of the sum of the two component variances |

All four arrays for a date have the same grid shape and can be opened with `numpy.load`. Readers can write their own plotting script to generate maps, comparisons, and uncertainty graphics from these arrays. The `.npy` files store array values; geographic coordinates are not embedded in them.

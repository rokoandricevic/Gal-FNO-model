import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# =========================================================================
# ⚙️ 1. CONFIGURATION
# =========================================================================
# --base-dir is the project directory, containing:
#   results/checkpoints/Gal-FNO_Fourier_Ensemble_{1..5}.pth
#   data/Bathymetry_20m.npy
#   data/GalFNO_Inference_Tensors_20m_21_23/*_10CH_*.bin
# The tensor directory can also be supplied with --tensors-dir.
# Four output maps per date are saved in results/inference/.

# These are the 20 m Chl target anchors in train_galfno.py (STATS["20m"]).
# Keep them in sync if the training targets or normalization are changed.
CHL_MEAN_20M = 0.7769
CHL_STD_20M = 0.3453

# Must match the continuous beta-mask construction used during training.
BETA_H_C = 3.5

TENSOR_FOLDER_NAME = "GalFNO_Inference_Tensors_20m_21_23"

# =========================================================================
# 📊 2. INPUT ROUTING & UTILS
# =========================================================================
def resolve_tensors_dir(data_dir, override):
    if override is not None:
        return os.path.abspath(os.path.expanduser(override))
    tensors_dir = os.path.join(data_dir, TENSOR_FOLDER_NAME)
    if not os.path.isdir(tensors_dir):
        raise FileNotFoundError(
            f"Inference tensor directory not found: {tensors_dir}. "
            "Extract the Zenodo archive there, or pass its location using --tensors-dir."
        )
    return tensors_dir

def create_2d_gaussian_window(window_h, window_w):
    y = np.linspace(-1, 1, window_h)
    x = np.linspace(-1, 1, window_w)
    y_grid, x_grid = np.meshgrid(y, x, indexing='ij')
    window = np.exp(-0.5 * (x_grid**2 + y_grid**2) / (0.33**2))
    return torch.tensor(window, dtype=torch.float32)

def get_chronological_tau_a(date_str):
    """Linearly interpolates tau_a chronologically between seasonal anchor RMSEs."""
    dt = datetime.strptime(date_str, "%Y%m%d")
    doy = dt.timetuple().tm_yday
    
    # Approx Day of Year for Anchors (Mid-Month)
    # Mar 15: 74 | Apr 15: 105 | Jun 15: 166 | Nov 15: 319
    doys = [74, 105, 166, 319]
    rmses = [0.035, 0.089, 0.21, 0.10]
    
    # Cyclical interpolation for dates crossing the winter boundary
    if doy < doys[0]:
        tau_a = np.interp(doy, [319 - 365, doys[0]], [0.10, 0.035])
    elif doy > doys[-1]:
        tau_a = np.interp(doy, [doys[-1], 74 + 365], [0.10, 0.035])
    else:
        tau_a = np.interp(doy, doys, rmses)
        
    return float(tau_a)

# =========================================================================
# 🧠 3. PURE-OPTICS GAL-FNO ARCHITECTURE
# =========================================================================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels; self.out_channels = out_channels
        self.modes1 = modes1; self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        m1, m2 = min(self.modes1, x.size(-2) // 2), min(self.modes2, x.size(-1) // 2 + 1)
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(
            x_ft[:, :, -m1:, :m2],
            self.weights2[:, :, -m1:, :m2]
        )
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

class Gal_FNO(nn.Module):
    def __init__(self, modes1=16, modes2=16, width=64):
        super(Gal_FNO, self).__init__()
        self.width = width
        self.fc0 = nn.Linear(10, self.width)
        self.conv0, self.conv1 = SpectralConv2d(self.width, self.width, modes1, modes2), SpectralConv2d(self.width, self.width, modes1, modes2)
        self.conv2, self.conv3 = SpectralConv2d(self.width, self.width, modes1, modes2), SpectralConv2d(self.width, self.width, modes1, modes2)
        self.w0, self.w1 = nn.Conv2d(self.width, self.width, 1), nn.Conv2d(self.width, self.width, 1)
        self.w2, self.w3 = nn.Conv2d(self.width, self.width, 1), nn.Conv2d(self.width, self.width, 1)
        self.fc1, self.fc2 = nn.Linear(self.width, 128), nn.Linear(128, 1)

    def forward(self, x, beta_mask):
        # Apply the same ex-ante bathymetric gate used during training.
        if beta_mask.dim() == 2:
            beta_mask = beta_mask.unsqueeze(0).unsqueeze(0)
        elif beta_mask.dim() == 3:
            beta_mask = beta_mask.unsqueeze(1)

        x = x * beta_mask

        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x).permute(0, 3, 1, 2)
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = F.gelu(self.conv3(x) + self.w3(x))
        x = x.permute(0, 2, 3, 1)
        return self.fc2(F.gelu(self.fc1(x))).permute(0, 3, 1, 2)


# =========================================================================
# 🚀 4. ENSEMBLE INFERENCE ENGINE
# =========================================================================
def run_ensemble_inference(base_dir, tensors_dir=None):
    data_dir = os.path.join(base_dir, "data")
    checkpoints_dir = os.path.join(base_dir, "results", "checkpoints")
    bathy_file = os.path.join(data_dir, "Bathymetry_20m.npy")
    out_dir = os.path.join(base_dir, "results", "inference")
    tensors_dir = resolve_tensors_dir(data_dir, tensors_dir)
    if not os.path.isfile(bathy_file):
        raise FileNotFoundError(f"Missing inference bathymetry: {bathy_file}")
    model_paths = [
        os.path.join(checkpoints_dir, f"Gal-FNO_Fourier_Ensemble_{i}.pth")
        for i in range(1, 6)
    ]
    missing_models = [path for path in model_paths if not os.path.isfile(path)]
    if missing_models:
        raise FileNotFoundError("Missing ensemble checkpoints:\n  " + "\n  ".join(missing_models))
    tensor_files = sorted(glob.glob(os.path.join(tensors_dir, "*_10CH_*.bin")))
    if not tensor_files:
        raise FileNotFoundError(f"No *_10CH_*.bin inference tensors found in {tensors_dir}")
    os.makedirs(out_dir, exist_ok=True)

    print("\n=========================================================================")
    print("      GAL-FNO: DEEP ENSEMBLE INFERENCE (5 MODELS + UNIFIED ALEATORY)")
    print("=========================================================================\n")
    global_chl_mean, global_chl_std = CHL_MEAN_20M, CHL_STD_20M
    print(f"   [Anchors] Using Global Target Mean: {global_chl_mean:.4f}, Std: {global_chl_std:.4f}")
    
    h_depth_master = np.load(bathy_file)
    if h_depth_master.ndim != 2:
        raise ValueError(
            f"Expected a 2-D bathymetry array, got shape {h_depth_master.shape}"
        )
    NY, bathy_nx = h_depth_master.shape
    
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"🖥️ Hardware Accelerator: {device.type.upper()}")

    # Construct one full-domain continuous beta gate and reuse it for all
    # overpasses. Invalid, land and non-positive bathymetry remain beta = 0.
    valid_depth = np.isfinite(h_depth_master) & (h_depth_master > 0.0)
    beta_master_np = np.zeros_like(h_depth_master, dtype=np.float32)
    beta_master_np[valid_depth] = (
        1.0
        - np.exp(
            -(h_depth_master[valid_depth] ** 2)
            / (2.0 * BETA_H_C ** 2)
        )
    )
    beta_master = torch.from_numpy(beta_master_np).to(device)
    print(
        f"   [Beta] Ex-ante gate ready: {NY} x {bathy_nx}; "
        f"nonzero pixels={np.count_nonzero(beta_master_np):,}; "
        f"range=[{beta_master_np.min():.4f}, {beta_master_np.max():.4f}]"
    )
    
    # Load all 5 models into a list
    ensemble_models = []
    print("\n🧠 Loading Deep Ensemble Master Brains...")
    for i, m_path in enumerate(model_paths, start=1):
        model = Gal_FNO().to(device)
        model.load_state_dict(torch.load(m_path, map_location=device, weights_only=True))
        model.eval()
        ensemble_models.append(model)
        print(f"   ✅ Loaded Member {i}/5")
    
    print(f"\n📦 Found {len(tensor_files)} pre-built pure-optics tensors. Initiating...\n")
    
    # 🌟 STRICT GEOMETRY PRESERVED
    patch_h, patch_w = 20, 60
    stride_h, stride_w = 5, 15
    gaussian_mask = create_2d_gaussian_window(patch_h, patch_w).to(device)
    
    for t_file in tensor_files:
        date_str = os.path.splitext(os.path.basename(t_file))[0].rsplit("_", 1)[-1]
        
        # 🌟 Explicit I/O routing for the Publication Plotting Script
        out_mean_path = os.path.join(out_dir, f"GalFNO_Mean_{date_str}.npy")
        out_alea_path = os.path.join(out_dir, f"GalFNO_AleatorySD_{date_str}.npy")
        out_epis_path = os.path.join(out_dir, f"GalFNO_EpistemicSD_{date_str}.npy")
        out_std_path  = os.path.join(out_dir, f"GalFNO_StdTotal_{date_str}.npy")
        
        print(f"🌀 Processing {date_str} (Ensemble Evaluation)...")
        raw_data = np.fromfile(t_file, dtype=np.float32)
        if raw_data.size % (NY * 10) != 0:
            raise ValueError(
                f"Tensor {t_file} contains {raw_data.size} values, which cannot "
                f"be reshaped into ({NY}, NX, 10)."
            )
        actual_nx = int(len(raw_data) / (NY * 10))
        if NY < 20 or actual_nx < 60:
            raise ValueError(f"Tensor {t_file} has grid {NY} x {actual_nx}; patches require at least 20 x 60.")
        if actual_nx > bathy_nx:
            raise ValueError(
                f"Tensor width {actual_nx} exceeds bathymetry width {bathy_nx}."
            )
        norm_tensor = raw_data.reshape(NY, actual_nx, 10)
        
        # --- Unified Aleatory Math & Chronological Routing ---
        tau_a = get_chronological_tau_a(date_str)
        Z_score = norm_tensor[:, :, 6]
        h_depth = h_depth_master[:, :actual_nx]
            
        W_min, A_min, h_c = 0.1, 0.1, 3.5
        W_bathy = 1.0 - (1.0 - W_min) * np.exp(-(h_depth**2) / (2 * h_c**2))
        C_hZ = A_min + (1.0 - A_min) * np.exp(-(Z_score**2) / 2.0) * W_bathy
        sigma_aleatory = tau_a * np.sqrt(1.0 / C_hZ)
        
        # --- Deep Ensemble Track (Batched GPU inference) ---
        full_tensor = torch.from_numpy(norm_tensor).to(device)
        beta_scene = beta_master[:, :actual_nx]
        
        # One full-domain accumulator per ensemble member. Each member must be
        # reconstructed first; ensemble variance is calculated only afterward.
        sum_member_preds = torch.zeros(
            (len(ensemble_models), NY, actual_nx),
            dtype=torch.float32,
            device=device,
        )
        sum_weights = torch.zeros((NY, actual_nx), device=device) + 1e-8
        
        spatial_batch = []
        coords_batch = []
        
        def process_batch(s_batch, c_batch):
            stacked = torch.stack(s_batch) # [Batch, 10, 20, 60]

            # Use the same patch coordinates for the optical tensor and beta.
            beta_batch = torch.stack([
                beta_scene[si:ei, sj:ej]
                for si, ei, sj, ej in c_batch
            ]) # [Batch, 20, 60]
            
            ensemble_preds = []
            for m in ensemble_models:
                preds = m(stacked, beta_batch).squeeze(1)
                ensemble_preds.append(preds)
                
            ensemble_stack = torch.stack(ensemble_preds, dim=0) # [5, Batch, 20, 60]

            for idx, (si, ei, sj, ej) in enumerate(c_batch):
                sum_member_preds[:, si:ei, sj:ej] += (
                    ensemble_stack[:, idx] * gaussian_mask.unsqueeze(0)
                )
                sum_weights[si:ei, sj:ej] += gaussian_mask

        with torch.no_grad():
            for i in range(0, NY - patch_h + stride_h, stride_h):
                for j in range(0, actual_nx - patch_w + stride_w, stride_w):
                    start_i = min(i, NY - patch_h)
                    start_j = min(j, actual_nx - patch_w)
                    end_i = start_i + patch_h
                    end_j = start_j + patch_w
                    
                    patch = full_tensor[start_i:end_i, start_j:end_j, :].permute(2, 0, 1)
                    spatial_batch.append(patch)
                    coords_batch.append((start_i, end_i, start_j, end_j))
                    
                    if len(spatial_batch) == 16:
                        process_batch(spatial_batch, coords_batch)
                        spatial_batch.clear()
                        coords_batch.clear()
                        
            if len(spatial_batch) > 0:
                process_batch(spatial_batch, coords_batch)
                spatial_batch.clear()
                coords_batch.clear()
                    
        # Normalize each member's accumulated map in place to limit memory use.
        sum_member_preds.div_(sum_weights.unsqueeze(0))

        # Correct order: Gaussian reconstruction first, ensemble statistics second.
        blended_mean = torch.mean(sum_member_preds, dim=0)
        blended_var = torch.var(sum_member_preds, dim=0, unbiased=True)

        del full_tensor, beta_scene, sum_member_preds, sum_weights
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        
        # --- Law of Total Variance Synthesis ---
        final_mean_phys = (blended_mean.cpu().numpy() * global_chl_std) + global_chl_mean
        final_mean_phys = np.clip(final_mean_phys, 0.0, None)
        
        sigma_epistemic = (torch.sqrt(blended_var).cpu().numpy() * global_chl_std)
        final_std_total = np.sqrt(sigma_aleatory**2 + sigma_epistemic**2)
        
        np.save(out_mean_path, final_mean_phys)
        np.save(out_alea_path, sigma_aleatory)
        np.save(out_epis_path, sigma_epistemic)
        np.save(out_std_path, final_std_total)
        
    print("\n🎉 SUCCESS: All ensemble inferences successfully executed and explicit components saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gal-FNO ensemble inference on 20 m basin tensors.")
    parser.add_argument(
        "--base-dir", default=os.path.dirname(os.path.abspath(__file__)),
        help="Project directory containing data/ and results/checkpoints/; defaults to the script's directory.",
    )
    parser.add_argument(
        "--tensors-dir", default=None,
        help="Extracted inference tensor directory; defaults to data/GalFNO_Inference_Tensors_20m_21_23/.",
    )
    args = parser.parse_args()
    run_ensemble_inference(os.path.abspath(os.path.expanduser(args.base_dir)), args.tensors_dir)

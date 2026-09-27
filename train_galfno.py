import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# =========================================================================
# ⚙️ 1. CONFIGURATION & DIRECTORY PATHS
# =========================================================================
# --base-dir is the project directory (defaults to the directory of this script).
# Required input files:
#   data/inputs_10m.bin, data/outputs_10m.bin
#   data/inputs_20m.bin, data/outputs_20m.bin
#   data/bathymetry/beta_mask_10m.bin, data/bathymetry/beta_mask_20m.bin
#   data/bathymetry/bathymetry_10m.bin, data/bathymetry/bathymetry_20m.bin
# Results are written beneath results/checkpoints and results/figures.


def project_directories(base_dir):
    data_dir = os.path.join(base_dir, "data")
    bathy_dir = os.path.join(data_dir, "bathymetry")
    checkpoints_dir = os.path.join(base_dir, "results", "checkpoints")
    figures_dir = os.path.join(base_dir, "results", "figures")
    eval_dir = os.path.join(figures_dir, "evaluation")
    return data_dir, bathy_dir, checkpoints_dir, figures_dir, eval_dir


def check_input_files(data_dir, bathy_dir):
    required = [
        os.path.join(data_dir, f"{kind}_{resolution}.bin")
        for resolution in ("10m", "20m")
        for kind in ("inputs", "outputs")
    ] + [
        os.path.join(bathy_dir, f"{kind}_{resolution}.bin")
        for resolution in ("10m", "20m")
        for kind in ("beta_mask", "bathymetry")
    ]
    missing = [path for path in required if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError("Required training files are missing:\n  " + "\n  ".join(missing))

# 🌟 STRICT BIO-OPTICAL REGULATION ENVELOPE 
REGRESSION_ENVELOPE = {
    "10m": {
        "June21":     {"a": 0.025, "b": 4.10, "rmse": 0.21}, 
        "November21": {"a": 0.20, "b": 1.30, "rmse": 0.10}, 
        "March22":    {"a": 0.15, "b": 1.20, "rmse": 0.034}, 
        "April22":    {"a": 0.0044, "b": 5.30, "rmse": 0.094}, 
    },
    "20m": {
        "June21":     {"a": 0.031, "b": 3.90, "rmse": 0.21}, 
        "November21": {"a": 0.20, "b": 1.30, "rmse": 0.10}, 
        "March22":    {"a": 0.15, "b": 1.20, "rmse": 0.035}, 
        "April22":    {"a": 0.0045, "b": 5.30, "rmse": 0.089}, 
    }
}

# 🌟 GLOBAL TARGET & R1 ANCHORS
STATS = {
    "10m": {
        "chl_mean": 0.7733, "chl_std": 0.3410,
        "r1_mean": 0.978893, "r1_std": 0.058000
    },
    "20m": {
        "chl_mean": 0.7769, "chl_std": 0.3453,
        "r1_mean": 0.978282, "r1_std": 0.056473
    }
}

SEASONS = ["June21", "November21", "March22", "April22"]
IDX_10M = [759, 1541, 2083, 3842]
IDX_20M = [997, 1619, 2437, 3835]
IDX_CROSS = [348, 1227, 2464, 3117] 

# =========================================================================
# 🌊 2. DATASET LOADER & DAFNO-STYLE GEOMETRY-ADAPTIVE ARCHITECTURE
# =========================================================================
class BinaryMeshDataset(torch.utils.data.Dataset):
    def __init__(self, inputs_path, outputs_path, nx, ny, num_channels=10, num_samples=4000):
        super(BinaryMeshDataset, self).__init__()
        self.num_samples = num_samples
        print(f" Loading Memory Mapping Stream -> {os.path.basename(inputs_path)}")
        self.inputs = np.fromfile(inputs_path, dtype=np.float32).reshape(num_samples, nx, ny, num_channels)
        self.outputs = np.fromfile(outputs_path, dtype=np.float32).reshape(num_samples, nx, ny)

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        x = torch.from_numpy(self.inputs[idx]).permute(2, 0, 1).contiguous()
        y = torch.from_numpy(self.outputs[idx]).unsqueeze(0).contiguous()
        return x, y, idx

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
        m1 = min(self.modes1, x.size(-2) // 2)
        m2 = min(self.modes2, x.size(-1) // 2 + 1)
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(x_ft[:, :, -m1:, :m2], self.weights2[:, :, -m1:, :m2])
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

class Gal_FNO(nn.Module):
    """
    Geometry adaptive learning Fourier Neural Operator (Gal-FNO).
    The spatial beta mask is applied directly to the input features inside 
    the forward pass prior to spectral convolutions, physically blinding 
    the Fourier layers to land boundaries and ultra-shallow optical noise.
    """
    def __init__(self, modes1=16, modes2=16, width=64):
        super(Gal_FNO, self).__init__()
        self.width = width
        self.fc0 = nn.Linear(10, self.width)
        self.conv0 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, beta_mask):
        # x shape: [batch, channels, H, W]
        # beta_mask shape: [H, W] or [batch, 1, H, W]
        if beta_mask.dim() == 2:
            beta_mask = beta_mask.unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
        elif beta_mask.dim() == 3:
            beta_mask = beta_mask.unsqueeze(1) # [batch, 1, H, W]
            
        # 🌟 EX-ANTE GEOMETRY MASKING (DAFNO MECHANISM)
        x = x * beta_mask

        x = x.permute(0, 2, 3, 1) # [batch, H, W, channels]
        x = self.fc0(x).permute(0, 3, 1, 2) # [batch, width, H, W]
        
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = F.gelu(self.conv3(x) + self.w3(x))
        
        x = x.permute(0, 2, 3, 1)
        return self.fc2(F.gelu(self.fc1(x))).permute(0, 3, 1, 2)

# =========================================================================
# 🛡️ 3. UNIFIED LOSS FUNCTION
# =========================================================================
def compute_unified_gal_pino_loss(preds, targets, inputs, beta_mask, pool_name, batch_indices, 
                                  chl_mean, chl_std, r1_mean, r1_std, 
                                  lambda_base=0.01, lambda_boundary=1.0):
    
    valid_area_per_slice = torch.sum(beta_mask) + 1e-8 
    
    preds_sq = preds.squeeze(1)
    targets_sq = targets.squeeze(1)
    
    data_sq_error = beta_mask * (preds_sq - targets_sq)**2
    data_loss = torch.mean(torch.sum(data_sq_error, dim=(1, 2)) / valid_area_per_slice)
    
    chl_pred_phys = (preds_sq * chl_std) + chl_mean
    ratio_z = inputs[:, 6, :, :]  # Channel 6 is R1 (B02/B03)
    ratio_phys = (ratio_z * r1_std) + r1_mean
    
    device = preds.device
    batch_size = preds.shape[0]
    
    a_mean = torch.zeros(batch_size, 1, 1, device=device)
    b_mean = torch.zeros(batch_size, 1, 1, device=device)
    rmse_val = torch.zeros(batch_size, 1, 1, device=device)
    
    season_keys = ["June21", "November21", "March22", "April22"]
    cfg = REGRESSION_ENVELOPE[pool_name]
    
    for b_idx in range(batch_size):
        global_idx = batch_indices[b_idx].item()
        season_idx = min(global_idx // 1000, 3) 
        season_key = season_keys[season_idx]
        
        a_mean[b_idx, 0, 0] = cfg[season_key]["a"]
        b_mean[b_idx, 0, 0] = cfg[season_key]["b"]
        rmse_val[b_idx, 0, 0] = cfg[season_key]["rmse"]
    
    mean_trend = a_mean * torch.exp(b_mean * ratio_phys)
    upper_bound = mean_trend + (4.0 * rmse_val)
    lower_bound = torch.clamp(mean_trend - (4.0 * rmse_val), min=0.0)
    
    base_sq_error = beta_mask * ((chl_pred_phys - mean_trend) / chl_std)**2
    base_loss = lambda_base * torch.mean(torch.sum(base_sq_error, dim=(1, 2)) / valid_area_per_slice)
    
    over_dist = torch.clamp(chl_pred_phys - upper_bound, min=0.0)
    under_dist = torch.clamp(lower_bound - chl_pred_phys, min=0.0)
    delta_z = (over_dist + under_dist) / chl_std
    
    boundary_sq_error = beta_mask * (delta_z**2)
    boundary_loss = lambda_boundary * torch.mean(torch.sum(boundary_sq_error, dim=(1, 2)) / valid_area_per_slice)
    
    total_loss = data_loss + base_loss + boundary_loss
    return total_loss, data_loss, base_loss, boundary_loss

# =========================================================================
# 🚀 4. MASTER TRAINING LOOP ENGINE (DEEP ENSEMBLE M=5)
# =========================================================================
def execute_gal_fno_fourier_training(base_dir):
    data_dir, bathy_dir, checkpoints_dir, figures_dir, eval_dir = project_directories(base_dir)
    check_input_files(data_dir, bathy_dir)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    print("\n=========================================================================")
    print("      INITIALIZING DAFNO ENSEMBLE: GAL-FNO FOURIER (5 MODELS)")
    print("=========================================================================")
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Assigning execution load straight to hardware device target: {device}\n")

    dataset_10m = BinaryMeshDataset(os.path.join(data_dir, "inputs_10m.bin"), os.path.join(data_dir, "outputs_10m.bin"), nx=40, ny=116)
    dataset_20m = BinaryMeshDataset(os.path.join(data_dir, "inputs_20m.bin"), os.path.join(data_dir, "outputs_20m.bin"), nx=20, ny=58)

    loader_10m = torch.utils.data.DataLoader(dataset_10m, batch_size=32, shuffle=True)
    loader_20m = torch.utils.data.DataLoader(dataset_20m, batch_size=32, shuffle=True)

    print("\nLoading Pre-Computed Gaussian Beta Masks...")
    beta_mask_10m = torch.tensor(np.fromfile(os.path.join(bathy_dir, "beta_mask_10m.bin"), dtype=np.float32).reshape(40, 116), device=device)
    beta_mask_20m = torch.tensor(np.fromfile(os.path.join(bathy_dir, "beta_mask_20m.bin"), dtype=np.float32).reshape(20, 58), device=device)

    penalty_boundary = 1.0  
    ensemble_loss_10m, ensemble_loss_20m = {}, {}

    # 🌟 DEEP ENSEMBLE LOOP (1 to 5)
    for ensemble_id in range(1, 6):
        print(f"\n=======================================================")
        print(f"   ▶️ INITIATING DAFNO ENSEMBLE MEMBER {ensemble_id} / 5")
        print(f"=======================================================")
        
        # Set manual seed for structural variance across ensemble members
        torch.manual_seed(41 + ensemble_id)
        model = Gal_FNO(modes1=16, modes2=16, width=64).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

        loss_history_10m, loss_history_20m = [], []

        for epoch in range(1, 51):
            model.train()
            accum_10m, accum_20m = 0.0, 0.0
            batches = 0
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                for (x10, y10, idx10), (x20, y20, idx20) in zip(loader_10m, loader_20m):
                    optimizer.zero_grad()
                    
                    # Pass the beta mask into the forward pass
                    pred10 = model(x10.to(device), beta_mask_10m)
                    pred20 = model(x20.to(device), beta_mask_20m)
                    
                    t10, _, _, _ = compute_unified_gal_pino_loss(
                        pred10, y10.to(device), x10.to(device), beta_mask_10m, "10m", idx10, 
                        STATS["10m"]["chl_mean"], STATS["10m"]["chl_std"], STATS["10m"]["r1_mean"], STATS["10m"]["r1_std"], 
                        lambda_boundary=penalty_boundary)
                    
                    t20, _, _, _ = compute_unified_gal_pino_loss(
                        pred20, y20.to(device), x20.to(device), beta_mask_20m, "20m", idx20, 
                        STATS["20m"]["chl_mean"], STATS["20m"]["chl_std"], STATS["20m"]["r1_mean"], STATS["20m"]["r1_std"], 
                        lambda_boundary=penalty_boundary)
                    
                    total_loss = t10 + t20
                    total_loss.backward()
                    optimizer.step()
                    
                    accum_10m += t10.item()
                    accum_20m += t20.item()
                    batches += 1
                    
            scheduler.step()
            loss_history_10m.append(accum_10m / batches)
            loss_history_20m.append(accum_20m / batches)
            
            if epoch % 10 == 0 or epoch == 1:
                print(f"   Epoch [{epoch:>2}/50] | Total Loss 10m: {loss_history_10m[-1]:.5f} | Total Loss 20m: {loss_history_20m[-1]:.5f}")

        ensemble_loss_10m[ensemble_id] = loss_history_10m
        ensemble_loss_20m[ensemble_id] = loss_history_20m
        
        # Keep the checkpoint filenames compatible with the existing inference script.
        weights_path = os.path.join(checkpoints_dir, f"Gal-FNO_Fourier_Ensemble_{ensemble_id}.pth")
        torch.save(model.state_dict(), weights_path)
        print(f"   ✅ Saved {os.path.basename(weights_path)}")

    print("\n🎉 SUCCESS: ALL 5 DAFNO ENSEMBLE MEMBERS SUCCESSFULLY TRAINED.")

    # 🌟 PLOT COMBINED LOSS CURVES
    plt.figure(figsize=(8, 5.5))
    for eid in range(1, 6):
        label_10 = 'Pool 10m' if eid == 1 else None
        label_20 = 'Pool 20m' if eid == 1 else None
        plt.plot(range(1, 51), ensemble_loss_10m[eid], label=label_10, color='royalblue', lw=1.5, alpha=0.7)
        plt.plot(range(1, 51), ensemble_loss_20m[eid], label=label_20, color='darkorange', lw=1.5, alpha=0.7)
        
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss (Beta-MSE + Stat-Bounds)')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.title("DAFNO Ensemble Convergence: Gal-FNO Fourier + Ex-Ante Masking")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "Gal-FNO_Fourier_Ensemble_Loss.png"), dpi=200)
    plt.close()
    
    # Trigger final evaluation rendering for the final model (Member 5)
    generate_evaluation_plates(model, beta_mask_10m, beta_mask_20m, device, data_dir, bathy_dir, eval_dir)

# =========================================================================
# 🎨 5. INTEGRATED EVALUATION RENDERER
# =========================================================================
def load_pool_data(data_dir, bathy_dir, pool_label, nx, ny):
    inputs = np.fromfile(os.path.join(data_dir, f"inputs_{pool_label}.bin"), dtype=np.float32).reshape(-1, nx, ny, 10)
    targets = np.fromfile(os.path.join(data_dir, f"outputs_{pool_label}.bin"), dtype=np.float32).reshape(-1, nx, ny)
    bathy = np.fromfile(os.path.join(bathy_dir, f"bathymetry_{pool_label}.bin"), dtype=np.float32).reshape(nx, ny)
    water_mask = (bathy > 0.001)
    return inputs, targets, water_mask

def generate_evaluation_plates(model, beta_10m, beta_20m, device, data_dir, bathy_dir, eval_dir):
    print("\n=========================================================================")
    print("      GAL-FNO FOURIER: GENERATING PUBLICATION EVALUATION PLATES")
    print("=========================================================================\n")
    model.eval()

    in_10, tgt_10, mask_10 = load_pool_data(data_dir, bathy_dir, "10m", 40, 116)
    in_20, tgt_20, mask_20 = load_pool_data(data_dir, bathy_dir, "20m", 20, 58)

    # -------------------------------------------------------------------------
    # PLATE 1: PHYSICS COMPLIANCE (10m & 20m)
    # -------------------------------------------------------------------------
    print("🎨 Rendering Bio-Optical Compliance Boundaries...")
    for p_label, inputs_arr, mask_arr, indices, beta_tensor in [("10m", in_10, mask_10, IDX_10M, beta_10m), ("20m", in_20, mask_20, IDX_20M, beta_20m)]:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i, (season, s_idx) in enumerate(zip(SEASONS, indices)):
            ax = axes[i]
            x_tensor = torch.tensor(inputs_arr[s_idx]).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_norm = model(x_tensor, beta_tensor).squeeze().cpu().numpy()
                
            chl_pred = (pred_norm * STATS[p_label]["chl_std"]) + STATS[p_label]["chl_mean"]
            r1_input = (inputs_arr[s_idx, :, :, 6] * STATS[p_label]["r1_std"]) + STATS[p_label]["r1_mean"]
            
            chl_valid = chl_pred[mask_arr].flatten()
            r1_valid = r1_input[mask_arr].flatten()
            
            ax.scatter(r1_valid, chl_valid, alpha=0.3, s=4, color='royalblue', label='Gal-FNO Predictions')
            
            a = REGRESSION_ENVELOPE[p_label][season]["a"]
            b = REGRESSION_ENVELOPE[p_label][season]["b"]
            rmse = REGRESSION_ENVELOPE[p_label][season]["rmse"]
            
            x_vals = np.linspace(np.min(r1_valid), np.max(r1_valid), 100)
            mean_trend = a * np.exp(b * x_vals)
            upper_bound = mean_trend + (4.0 * rmse)
            lower_bound = np.clip(mean_trend - (4.0 * rmse), 0.0, None)
            
            ax.plot(x_vals, mean_trend, 'k--', lw=2.5, label='Bio-Optical Mean Trend')
            ax.plot(x_vals, upper_bound, 'r-', lw=2, label='Upper Bound')
            ax.plot(x_vals, lower_bound, 'r-', lw=2, label='Lower Bound')
            
            ax.set_title(f"{season} Boundary Compliance", fontsize=13, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.6)
            if i >= 2: ax.set_xlabel("Optical Ratio (B02/B03)", fontsize=11)
            if i % 2 == 0: ax.set_ylabel("Chlorophyll-a [mg/m³]", fontsize=11)
            if i == 0: ax.legend(loc='upper left', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(eval_dir, f"Gal-FNO_Fourier_Physics_Compliance_{p_label}.png"), dpi=300)
        plt.close()

    # -------------------------------------------------------------------------
    # PLATE 2: SPATIAL GRID COMPARISON (10m & 20m)
    # -------------------------------------------------------------------------
    print("🎨 Rendering Spatial Grid Target vs Predictions...")
    for p_label, inputs_arr, tgt_arr, mask_arr, indices, beta_tensor in [("10m", in_10, tgt_10, mask_10, IDX_10M, beta_10m), ("20m", in_20, tgt_20, mask_20, IDX_20M, beta_20m)]:
        fig, axes = plt.subplots(4, 3, figsize=(18, 12))
        
        for i, (season, s_idx) in enumerate(zip(SEASONS, indices)):
            x_tensor = torch.tensor(inputs_arr[s_idx]).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_norm = model(x_tensor, beta_tensor).squeeze().cpu().numpy()
                
            pred_phys = (pred_norm * STATS[p_label]["chl_std"]) + STATS[p_label]["chl_mean"]
            tgt_phys = (tgt_arr[s_idx] * STATS[p_label]["chl_std"]) + STATS[p_label]["chl_mean"]
            
            pred_masked = np.where(mask_arr, pred_phys, np.nan)
            tgt_masked = np.where(mask_arr, tgt_phys, np.nan)
            abs_err = np.abs(pred_masked - tgt_masked)
            
            vmax_chl = np.nanpercentile(tgt_masked, 98)
            vmax_err = np.nanpercentile(abs_err, 98)
            
            im0 = axes[i, 0].imshow(tgt_masked, cmap='viridis', aspect='auto', vmin=0, vmax=vmax_chl)
            axes[i, 0].set_title(f"{season} Target (Realization #{s_idx})", fontsize=12, fontweight='bold')
            axes[i, 0].axis('off')
            fig.colorbar(im0, ax=axes[i, 0], orientation='horizontal', fraction=0.05, pad=0.05, label="Chl-a (mg/m³)")
            
            im1 = axes[i, 1].imshow(pred_masked, cmap='viridis', aspect='auto', vmin=0, vmax=vmax_chl)
            axes[i, 1].set_title(f"Gal-FNO Prediction", fontsize=12, fontweight='bold')
            axes[i, 1].axis('off')
            fig.colorbar(im1, ax=axes[i, 1], orientation='horizontal', fraction=0.05, pad=0.05, label="Chl-a (mg/m³)")
            
            im2 = axes[i, 2].imshow(abs_err, cmap='magma', aspect='auto', vmin=0, vmax=vmax_err)
            axes[i, 2].set_title(f"Absolute Error", fontsize=12, fontweight='bold')
            axes[i, 2].axis('off')
            fig.colorbar(im2, ax=axes[i, 2], orientation='horizontal', fraction=0.05, pad=0.05, label="Error (mg/m³)")

        plt.tight_layout()
        plt.savefig(os.path.join(eval_dir, f"Gal-FNO_Fourier_Spatial_Grid_{p_label}.png"), dpi=300)
        plt.close()

    # -------------------------------------------------------------------------
    # PLATE 3: CROSS-RESOLUTION ZOOM (10m vs 20m)
    # -------------------------------------------------------------------------
    print("🎨 Rendering Cross-Resolution Stability Maps...")
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    
    for i, (season, s_idx) in enumerate(zip(SEASONS, IDX_CROSS)):
        x10 = torch.tensor(in_10[s_idx]).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad(): pred10_norm = model(x10, beta_10m).squeeze().cpu().numpy()
        pred10_phys = (pred10_norm * STATS["10m"]["chl_std"]) + STATS["10m"]["chl_mean"]
        pred10_masked = np.where(mask_10, pred10_phys, np.nan)
        
        x20 = torch.tensor(in_20[s_idx]).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad(): pred20_norm = model(x20, beta_20m).squeeze().cpu().numpy()
        pred20_phys = (pred20_norm * STATS["20m"]["chl_std"]) + STATS["20m"]["chl_mean"]
        pred20_masked = np.where(mask_20, pred20_phys, np.nan)
        
        vmax = max(np.nanpercentile(pred10_masked, 98), np.nanpercentile(pred20_masked, 98))
        
        im0 = axes[i, 0].imshow(pred10_masked, cmap='viridis', aspect='auto', vmin=0, vmax=vmax)
        axes[i, 0].set_ylabel(f"{season}\n(Realization #{s_idx})", fontsize=12, fontweight='bold')
        axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])
        if i == 0: axes[i, 0].set_title("10m Prediction", fontsize=13, fontweight='bold')
        fig.colorbar(im0, ax=axes[i, 0], orientation='horizontal', fraction=0.06, pad=0.05, label="Chl-a (mg/m³)")
        
        im1 = axes[i, 1].imshow(pred20_masked, cmap='viridis', aspect='auto', vmin=0, vmax=vmax)
        axes[i, 1].set_xticks([]); axes[i, 1].set_yticks([])
        if i == 0: axes[i, 1].set_title("20m Prediction", fontsize=13, fontweight='bold')
        fig.colorbar(im1, ax=axes[i, 1], orientation='horizontal', fraction=0.06, pad=0.05, label="Chl-a (mg/m³)")

    plt.suptitle("Dual Resolution Stability (10m vs 20m)", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(eval_dir, "Gal-FNO_Fourier_Cross_Resolution_Zoom.png"), dpi=300)
    plt.close()

    print(f"🎉 SUCCESS: All 3 DAFNO validation plates successfully generated in {eval_dir}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Gal-FNO ensemble on 10 m and 20 m data.")
    parser.add_argument(
        "--base-dir", default=os.path.dirname(os.path.abspath(__file__)),
        help="Project directory containing data/; defaults to the script's directory.",
    )
    args = parser.parse_args()
    execute_gal_fno_fourier_training(os.path.abspath(os.path.expanduser(args.base_dir)))

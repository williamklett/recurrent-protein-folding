
import os
import numpy as np
import torch
import csv
import json
import logging
from pathlib import Path
from transformers import AutoTokenizer, EsmForProteinFolding
from psa.models import PSA
from psa.targets_k2 import K_PAIR_K2

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("hard_svd_rounding")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifest.csv"
PHYS_DIR = ROOT / "data/physics"
RUNS_DIR = ROOT / "runs"

TEST_BLOCKS = [25, 43] # Peak Relaxation vs. Peak Crystallization
SEED = 0

def load_psa(block_idx: int, device: str) -> PSA:
    block_dir = RUNS_DIR / f"k2_block_{block_idx:02d}"
    psa = PSA(d_pair=128, d_hidden=128, K=K_PAIR_K2)
    psa.load_state_dict(torch.load(block_dir / "model.pt", map_location="cpu", weights_only=True))
    psa.eval()
    return psa.to(device)

def double_center(D2):
    row_means = D2.mean(dim=-1, keepdim=True)
    col_means = D2.mean(dim=-2, keepdim=True)
    total_mean = D2.mean(dim=(-1, -2), keepdim=True)
    return -0.5 * (D2 - row_means - col_means + total_mean)

def apply_hard_svd(dist_flat, L):
    """
    Apply Hard SVD Projection to a flat distance vector.
    dist_flat: (L*L, 1) or similar.
    """
    D = dist_flat.view(L, L)
    D2 = D ** 2
    G = double_center(D2)
    
    # Eigendecomposition (G is symmetric)
    # L is usually small enough for eigh on CPU/GPU
    eigvals, eigvecs = torch.linalg.eigh(G)
    
    # Hard SVD: Keep only top 3 positive eigenvalues
    # eigvals are sorted ascending by eigh
    new_eigvals = torch.zeros_like(eigvals)
    # Top 3 are at the end
    indices = torch.arange(len(eigvals) - 3, len(eigvals), device=eigvals.device)
    # Only keep if positive
    mask = eigvals[indices] > 0
    new_eigvals[indices[mask]] = eigvals[indices[mask]]
    
    # Reconstruct Gram matrix
    G_proj = eigvecs @ torch.diag(new_eigvals) @ eigvecs.T
    
    # Back to distances: D^2 = Gii + Gjj - 2Gij
    diag = torch.diag(G_proj).view(L, 1)
    D2_proj = diag + diag.T - 2 * G_proj
    D_proj = torch.sqrt(torch.clamp(D2_proj, min=0))
    
    return D_proj.view(-1, 1)

def make_hard_svd_hook(psa: PSA, apply_svd=True):
    def hook(module, args, output):
        s, z = output
        L = z.shape[2]
        z_flat = z[0].reshape(L * L, 128).to(next(psa.parameters()).dtype)
        
        with torch.no_grad():
            # pv maps to (dist, orient_cos)
            z_phys = psa.pv(z_flat) # (L*L, 2)
            
            if apply_svd:
                # Perform Hard SVD on the distance channel (index 0)
                d_orig = z_phys[:, 0:1]
                d_proj = apply_hard_svd(d_orig, L)
                
                # Combine with original orientation (index 1)
                z_phys_new = torch.cat([d_proj, z_phys[:, 1:2]], dim=-1)
            else:
                z_phys_new = z_phys
                
            # pr reconstructs to latent space
            z_recon = psa.pr(z_phys_new)
            
        z_new = z_recon.reshape(1, L, L, 128).to(z.dtype)
        return (s, z_new)
    return hook

def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True).to(device).eval()
    model.esm = model.esm.half()
    
    # Sample val proteins
    with open(MANIFEST) as f:
        proteins = [r for r in csv.DictReader(f) if (PHYS_DIR / f"{r['pdb_id'].lower()}.npz").exists()]
    
    # Just take 3 proteins to test the hypothesis (1TIT, 1LMB, 1LYZ)
    test_ids = ["1tit", "1lmb", "1lyz"]
    test_set = [p for p in proteins if p['pdb_id'].lower() in test_ids]
    
    results = []
    
    # Conditions: baseline, block25-PSA, block25-HardSVD, block43-PSA, block43-HardSVD
    for block in TEST_BLOCKS:
        psa = load_psa(block, device)
        
        for mode in ["psa_only", "hard_svd"]:
            log.info(f"Condition: Block {block}, Mode: {mode}")
            apply_svd = (mode == "hard_svd")
            handle = model.trunk.blocks[block].register_forward_hook(make_hard_svd_hook(psa, apply_svd))
            
            for p in test_set:
                phys = np.load(PHYS_DIR / f"{p['pdb_id'].lower()}.npz")
                gt_dist = phys["distances"].astype(np.float32)
                
                ids = tok([str(phys["sequence"])], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
                with torch.no_grad():
                    out = model(ids)
                
                pos = out["positions"][-1, 0]
                ca = pos[:, 1, :]
                diff = ca[:, None, :] - ca[None, :, :]
                pred_d = (diff ** 2).sum(-1).sqrt().cpu().numpy()
                plddt = out["plddt"][0].float().cpu().numpy().mean()
                
                mae = float(np.abs(pred_d - gt_dist).mean())
                results.append({
                    "block": block,
                    "mode": mode,
                    "pdb_id": p['pdb_id'],
                    "mae": mae,
                    "plddt": float(plddt)
                })
                log.info(f"  {p['pdb_id']}: MAE={mae:.3f}, pLDDT={plddt:.3f}")
                
            handle.remove()

    # Save
    with open(RUNS_DIR / "hard_svd_rounding_test.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()

import torch
import os
from pathlib import Path
import numpy as np

def compare_tensors(file1, file2, name):
    if not file1.exists():
        print(f"Missing {file1}")
        return
    if not file2.exists():
        print(f"Missing {file2}")
        return

    t1 = torch.load(file1, map_location="cpu")
    t2 = torch.load(file2, map_location="cpu")

    if t1.shape != t2.shape:
        print(f"[{name}] Shape mismatch: {t1.shape} vs {t2.shape}")
        return

    diff = (t1 - t2).abs()
    mae = diff.mean().item()
    max_diff = diff.max().item()
    std_diff = diff.std().item()
    
    t1_mean = t1.abs().mean().item()
    t2_mean = t2.abs().mean().item()

    print(f"--- {name} ---")
    print(f"  Shape: {t1.shape}")
    print(f"  Clean Mean Abs: {t1_mean:.6f}")
    print(f"  LoRA  Mean Abs: {t2_mean:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  Max Diff: {max_diff:.6f}")
    print(f"  Std Diff: {std_diff:.6f}")
    
    if mae < 1e-6:
        print("  >> IDENTICAL")
    else:
        print("  >> DIVERGED")
    print("")

def main():
    clean_dir = Path("debug_latents_clean")
    lora_dir = Path("debug_latents")

    print("Comparing Clean vs LoRA Latents...\n")

    # Compare Init
    compare_tensors(clean_dir / "latents_init.pt", lora_dir / "latents_init.pt", "Initial Latents")

    # Compare Steps
    steps = [1, 5, 8]
    for step in steps:
        fname = f"latents_step_{step}.pt"
        compare_tensors(clean_dir / fname, lora_dir / fname, f"Step {step}")

if __name__ == "__main__":
    main()

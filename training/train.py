"""
SNN vs ANN — Legitimate training comparison on 17×17 MNIST
===========================================================

Architecture (identical across all three models):
  Input:  17×17 = 289 neurons
  Hidden: 256   (fully connected)
  Output: 10    (fully connected, one per digit class)

Models:
  ANN          — PyTorch, ReLU activations, cross-entropy loss
  SNN-Rate     — snnTorch, LIF neurons, Poisson input encoding
  SNN-Temporal — snnTorch, LIF neurons, time-to-first-spike (TTFS) encoding

What this measures (all from real inference, nothing estimated):
  - Accuracy, precision, recall, F1 per class
  - Actual spike rates per layer (measured, not assumed)
  - Synaptic operations (SOPs) counted from real spike tensors
  - Energy via Horowitz 2014 MAC/AC model applied to real op counts
  - R/W memory operations on digital hardware (shows SNN overhead)

Output: results.json  (loadable by the React dashboard)
"""

import time
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate, functional as SF
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import numpy as np

# ─── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE   = 17
N_IN       = IMG_SIZE * IMG_SIZE   # 289
N_HID      = 256
N_OUT      = 10
T          = 25          # SNN timesteps per sample (rate window / TTFS window)
BATCH      = 128
EPOCHS     = 20
LR         = 1e-3
BETA       = 0.9         # LIF membrane decay constant
SPIKE_GRAD = surrogate.fast_sigmoid(slope=25)

# Horowitz 2014, 45 nm CMOS — used identically to the dashboard formula
E_MAC = 4.6e-12   # J — multiply-accumulate (ANN dense ops)
E_AC  = 0.9e-12   # J — accumulate only     (SNN spike-triggered add)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Precomputed ANN MAC counts (exact, from layer shapes)
FC1_MACS = N_IN  * N_HID   # 289 × 256 = 73,984
FC2_MACS = N_HID * N_OUT   # 256 × 10  =  2,560
ANN_TOTAL_MACS = FC1_MACS + FC2_MACS   # 76,544


# ─── Data ──────────────────────────────────────────────────────────────────────
def get_loaders():
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        # No normalisation: keep pixel values in [0,1] for spike encoding
    ])
    train_set = datasets.MNIST("data", train=True,  download=True, transform=tf)
    test_set  = datasets.MNIST("data", train=False, download=True, transform=tf)
    return (
        DataLoader(train_set, batch_size=BATCH, shuffle=True,  num_workers=2, pin_memory=True),
        DataLoader(test_set,  batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True),
    )


# ─── Input encodings ───────────────────────────────────────────────────────────
def rate_encode(x: torch.Tensor, T: int) -> torch.Tensor:
    """
    Poisson rate encoding.
    Pixel intensity p ∈ [0,1] → spike probability p per timestep.
    White pixel (p=1) spikes every step; black pixel (p=0) never spikes.

    x:      [B, 1, H, W]
    return: [T, B, N_IN]
    """
    x_flat = x.view(x.size(0), -1)                           # [B, N_IN]
    t_range = x_flat.unsqueeze(0).expand(T, -1, -1)           # [T, B, N_IN]
    return torch.bernoulli(t_range)


def ttfs_encode(x: torch.Tensor, T: int, threshold: float = 0.02) -> torch.Tensor:
    """
    Time-to-first-spike (TTFS) encoding.
    Strong pixel → early spike. Weak pixel (< threshold) → no spike.

    spike_time = round((1 - intensity) * (T-1))
      intensity=1.0 → t=0  (fires immediately)
      intensity=0.5 → t=12 (fires at midpoint)
      intensity=0.0 → no spike (masked out)

    x:      [B, 1, H, W]
    return: [T, B, N_IN]
    """
    x_flat  = x.view(x.size(0), -1)                          # [B, N_IN]
    active  = x_flat > threshold                              # [B, N_IN] bool
    spike_t = ((1.0 - x_flat) * (T - 1)).long().clamp(0, T - 1)  # [B, N_IN]

    t_range = torch.arange(T, device=x.device).view(T, 1, 1)
    spikes  = (spike_t.unsqueeze(0) == t_range).float()      # [T, B, N_IN]
    return spikes * active.unsqueeze(0).float()               # mask inactive pixels


# ─── Models ────────────────────────────────────────────────────────────────────
class ANN(nn.Module):
    """Standard MLP: 289 → ReLU(256) → 10"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N_IN, N_HID)
        self.fc2 = nn.Linear(N_HID, N_OUT)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc2(F.relu(self.fc1(x)))


class SNN(nn.Module):
    """
    LIF spiking network: 289 → LIF(256) → LIF(10)
    Surrogate gradient (fast sigmoid) enables backprop through spike events.
    Same weight count as ANN — only the activation function differs.
    """
    def __init__(self):
        super().__init__()
        self.fc1  = nn.Linear(N_IN, N_HID)
        self.lif1 = snn.Leaky(beta=BETA, spike_grad=SPIKE_GRAD)
        self.fc2  = nn.Linear(N_HID, N_OUT)
        self.lif2 = snn.Leaky(beta=BETA, spike_grad=SPIKE_GRAD)

    def forward(self, spk_in: torch.Tensor):
        """
        spk_in: [T, B, N_IN]  — pre-encoded spike train
        returns:
          spk2_rec: [T, B, N_OUT]  — output spike trains
          spk1_rec: [T, B, N_HID]  — hidden spike trains (for spike rate measurement)
        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk1_rec, spk2_rec = [], []

        for t in range(spk_in.size(0)):
            cur1 = self.fc1(spk_in[t])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk1_rec.append(spk1)
            spk2_rec.append(spk2)

        return torch.stack(spk2_rec, dim=0), torch.stack(spk1_rec, dim=0)


# ─── Operation counters ────────────────────────────────────────────────────────
def ann_ops() -> dict:
    """
    ANN: every weight fires every inference.
    R/W model: each weight must be read once; activations written and read.
    """
    energy_J  = ANN_TOTAL_MACS * E_MAC
    # Memory ops (floats): input + fc1 weights + hidden activations + fc2 weights + output
    reads     = N_IN + FC1_MACS + N_HID + FC2_MACS
    writes    = N_HID + N_OUT
    return {
        "macs":        ANN_TOTAL_MACS,
        "energy_pJ":   round(energy_J * 1e12, 2),
        "reads":       reads,
        "writes":      writes,
        "rw_total":    reads + writes,
    }


def snn_ops(spk_in: torch.Tensor, spk1: torch.Tensor, spk2: torch.Tensor,
             ann_energy_pJ: float) -> dict:
    """
    Count real SOPs from measured spike tensors.

    spk_in: [T, N_test, N_IN]   — input spike trains
    spk1:   [T, N_test, N_HID]  — hidden layer spikes
    spk2:   [T, N_test, N_OUT]  — output layer spikes

    SOPs per sample (total over all T timesteps):
      fc1 SOPs = Σ_t  input_spikes_at_t  × N_HID   (each input spike fans out to N_HID weights)
      fc2 SOPs = Σ_t  hidden_spikes_at_t × N_OUT

    R/W on digital hardware (important: membrane state must be maintained every step):
      Every timestep:
        - Read  all membrane potentials (N_HID + N_OUT)
        - Write all membrane potentials (N_HID + N_OUT)
        - Read  weights only for spiking pre-synaptic neurons
      This means SNN has MORE memory traffic than ANN on digital hardware —
      the efficiency argument only holds on event-driven neuromorphic chips.
    """
    # Mean per-sample spike counts (summed over T)
    in_spikes_per_sample  = spk_in.sum(dim=[0, 2]).mean().item()   # total input spikes
    hid_spikes_per_sample = spk1.sum(dim=[0, 2]).mean().item()     # total hidden spikes
    out_spikes_per_sample = spk2.sum(dim=[0, 2]).mean().item()

    # Spike rates (fraction of neurons active per timestep, averaged over T and samples)
    r_in  = spk_in.mean().item()
    r_hid = spk1.mean().item()
    r_out = spk2.mean().item()

    # Synaptic ops (additions only — no multiply)
    sops_fc1 = in_spikes_per_sample  * N_HID
    sops_fc2 = hid_spikes_per_sample * N_OUT
    total_sops = sops_fc1 + sops_fc2

    energy_pJ = total_sops * E_AC * 1e12

    # R/W on digital hardware
    # Membrane reads/writes: every timestep for every neuron regardless of spikes
    rw_membrane = T * (N_HID + N_OUT) * 2         # ×2 for read+write
    # Weight reads: only when pre-synaptic neuron spikes
    weight_reads_fc1 = in_spikes_per_sample  * N_HID   # per sample over T
    weight_reads_fc2 = hid_spikes_per_sample * N_OUT
    reads  = rw_membrane / 2 + weight_reads_fc1 + weight_reads_fc2
    writes = rw_membrane / 2
    rw_total = reads + writes

    return {
        "total_sops":         round(total_sops),
        "sops_fc1":           round(sops_fc1),
        "sops_fc2":           round(sops_fc2),
        "energy_pJ":          round(energy_pJ, 2),
        "energy_gain_vs_ann": round(ann_energy_pJ / energy_pJ, 2),
        "r_input":            round(r_in  * 100, 2),   # %
        "r_hidden":           round(r_hid * 100, 2),
        "r_output":           round(r_out * 100, 2),
        "in_spikes_per_sample":  round(in_spikes_per_sample,  1),
        "hid_spikes_per_sample": round(hid_spikes_per_sample, 1),
        "out_spikes_per_sample": round(out_spikes_per_sample, 1),
        "reads":              round(reads),
        "writes":             round(writes),
        "rw_total":           round(rw_total),
        "rw_vs_ann":          round(rw_total / (ann_ops()["rw_total"]), 2),
    }


# ─── Training loops ────────────────────────────────────────────────────────────
def train_ann_epoch(model, loader, optimiser, criterion):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimiser.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * x.size(0)
        correct    += (out.argmax(1) == y).sum().item()
        n          += x.size(0)
    return total_loss / n, correct / n * 100


def train_snn_epoch(model, loader, optimiser, encode_fn):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    loss_fn = SF.ce_rate_loss()
    for x, y in loader:
        x, y   = x.to(DEVICE), y.to(DEVICE)
        spk_in = encode_fn(x, T)
        optimiser.zero_grad()
        spk_out, _ = model(spk_in)
        loss = loss_fn(spk_out, y)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * x.size(0)
        correct    += SF.accuracy_rate(spk_out, y).item() * x.size(0)
        n          += x.size(0)
    return total_loss / n, correct / n * 100


# ─── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_ann(model, loader):
    model.eval()
    preds_all, labels_all = [], []
    for x, y in loader:
        preds_all.append(model(x.to(DEVICE)).argmax(1).cpu())
        labels_all.append(y)
    preds  = torch.cat(preds_all).numpy()
    labels = torch.cat(labels_all).numpy()
    return classification_report(labels, preds, output_dict=True, zero_division=0)


@torch.no_grad()
def eval_snn(model, loader, encode_fn, ann_energy_pJ: float):
    model.eval()
    preds_all, labels_all     = [], []
    spk_in_all, spk1_all, spk2_all = [], [], []

    for x, y in loader:
        x   = x.to(DEVICE)
        spk_in  = encode_fn(x, T)
        spk_out, spk_hid = model(spk_in)
        preds_all.append(spk_out.sum(0).argmax(1).cpu())
        labels_all.append(y)
        spk_in_all.append(spk_in.cpu())
        spk1_all.append(spk_hid.cpu())
        spk2_all.append(spk_out.cpu())

    preds  = torch.cat(preds_all).numpy()
    labels = torch.cat(labels_all).numpy()
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    ops    = snn_ops(
        torch.cat(spk_in_all, dim=1),
        torch.cat(spk1_all,   dim=1),
        torch.cat(spk2_all,   dim=1),
        ann_energy_pJ,
    )
    return report, ops


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print(f"Architecture: {N_IN} → {N_HID} → {N_OUT}  |  T={T} timesteps  |  {EPOCHS} epochs")
    print(f"ANN MACs: {ANN_TOTAL_MACS:,}  ({FC1_MACS:,} fc1 + {FC2_MACS:,} fc2)")
    print()

    train_loader, test_loader = get_loaders()
    results   = {}
    ann_e_pJ  = ann_ops()["energy_pJ"]

    # ── ANN ──────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("ANN  (PyTorch, ReLU, cross-entropy)")
    print("=" * 60)
    model    = ANN().to(DEVICE)
    opt      = torch.optim.Adam(model.parameters(), lr=LR)
    crit     = nn.CrossEntropyLoss()
    history  = []

    for ep in range(EPOCHS):
        t0 = time.time()
        loss, acc = train_ann_epoch(model, train_loader, opt, crit)
        history.append({"epoch": ep + 1, "loss": round(loss, 4), "acc": round(acc, 2)})
        print(f"  ep {ep+1:2d}  loss={loss:.4f}  train_acc={acc:.1f}%  ({time.time()-t0:.1f}s)")

    report = eval_ann(model, test_loader)
    ops    = ann_ops()
    results["ann"] = {
        "history":    history,
        "final_acc":  round(report["accuracy"] * 100, 2),
        "report":     report,
        "ops":        ops,
    }
    print(f"\n  Test accuracy : {results['ann']['final_acc']}%")
    print(f"  MACs          : {ops['macs']:,}")
    print(f"  Energy        : {ops['energy_pJ']} pJ/inference")
    print(f"  R/W ops       : {ops['rw_total']:,}")

    # ── SNN-Rate ─────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("SNN-Rate  (snnTorch, LIF, Poisson encoding)")
    print("=" * 60)
    model   = SNN().to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    history = []

    for ep in range(EPOCHS):
        t0 = time.time()
        loss, acc = train_snn_epoch(model, train_loader, opt, rate_encode)
        history.append({"epoch": ep + 1, "loss": round(loss, 4), "acc": round(acc, 2)})
        print(f"  ep {ep+1:2d}  loss={loss:.4f}  train_acc={acc:.1f}%  ({time.time()-t0:.1f}s)")

    report, ops = eval_snn(model, test_loader, rate_encode, ann_e_pJ)
    results["snn_rate"] = {
        "history":    history,
        "final_acc":  round(report["accuracy"] * 100, 2),
        "report":     report,
        "ops":        ops,
    }
    print(f"\n  Test accuracy : {results['snn_rate']['final_acc']}%")
    print(f"  SOPs          : {ops['total_sops']:,}  (fc1: {ops['sops_fc1']:,}  fc2: {ops['sops_fc2']:,})")
    print(f"  Energy        : {ops['energy_pJ']} pJ  →  {ops['energy_gain_vs_ann']}× gain vs ANN")
    print(f"  Spike rates   : input {ops['r_input']}%  hidden {ops['r_hidden']}%  output {ops['r_output']}%")
    print(f"  R/W ops       : {ops['rw_total']:,}  ({ops['rw_vs_ann']}× vs ANN)")

    # ── SNN-Temporal ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("SNN-Temporal  (snnTorch, LIF, TTFS encoding)")
    print("=" * 60)
    model   = SNN().to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    history = []

    for ep in range(EPOCHS):
        t0 = time.time()
        loss, acc = train_snn_epoch(model, train_loader, opt, ttfs_encode)
        history.append({"epoch": ep + 1, "loss": round(loss, 4), "acc": round(acc, 2)})
        print(f"  ep {ep+1:2d}  loss={loss:.4f}  train_acc={acc:.1f}%  ({time.time()-t0:.1f}s)")

    report, ops = eval_snn(model, test_loader, ttfs_encode, ann_e_pJ)
    results["snn_temporal"] = {
        "history":    history,
        "final_acc":  round(report["accuracy"] * 100, 2),
        "report":     report,
        "ops":        ops,
    }
    print(f"\n  Test accuracy : {results['snn_temporal']['final_acc']}%")
    print(f"  SOPs          : {ops['total_sops']:,}  (fc1: {ops['sops_fc1']:,}  fc2: {ops['sops_fc2']:,})")
    print(f"  Energy        : {ops['energy_pJ']} pJ  →  {ops['energy_gain_vs_ann']}× gain vs ANN")
    print(f"  Spike rates   : input {ops['r_input']}%  hidden {ops['r_hidden']}%  output {ops['r_output']}%")
    print(f"  R/W ops       : {ops['rw_total']:,}  ({ops['rw_vs_ann']}× vs ANN)")

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<18} {'Acc':>7} {'Ops':>10} {'Energy pJ':>12} {'Gain':>7} {'R/W':>10} {'R/W×ANN':>9}")
    print("-" * 75)

    ann_r = results["ann"]
    print(f"{'ANN':<18} {ann_r['final_acc']:>6.2f}%"
          f" {ann_r['ops']['macs']:>10,}"
          f" {ann_r['ops']['energy_pJ']:>12.1f}"
          f" {'—':>7}"
          f" {ann_r['ops']['rw_total']:>10,}"
          f" {'1.00×':>9}")

    for key, label in [("snn_rate", "SNN-Rate"), ("snn_temporal", "SNN-Temporal")]:
        r   = results[key]
        ops = r["ops"]
        print(f"{label:<18} {r['final_acc']:>6.2f}%"
              f" {ops['total_sops']:>10,}"
              f" {ops['energy_pJ']:>12.1f}"
              f" {ops['energy_gain_vs_ann']:>6.2f}×"
              f" {ops['rw_total']:>10,}"
              f" {ops['rw_vs_ann']:>8.2f}×")

    print()
    print("Note: R/W > 1× means SNN uses MORE memory traffic than ANN on digital hardware.")
    print("      Energy < 1× (gain > 1×) means SNN uses less compute energy.")
    print("      These can diverge because membrane state must be updated every timestep")
    print("      regardless of spikes — the efficiency argument holds on neuromorphic chips.")

    # ── Save ─────────────────────────────────────────────────────────────────
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved → results.json")


if __name__ == "__main__":
    main()

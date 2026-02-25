# GCBF+ PyTorch Reimplementation

A from-scratch PyTorch reimplementation of the **GCBF+** algorithm from the paper *"GCBF+: A Neural Graph Control Barrier Function Framework for Distributed Safe Multi-Agent Control"*.

## Project Structure

```
gcbf_plus/
├── env/
│   └── double_integrator.py   # 2D multi-agent Double Integrator dynamics
├── nn/
│   ├── mlp.py                 # Generic MLP building block
│   ├── gnn.py                 # GNN layer with attention aggregation (Eq. 18)
│   ├── gcbf_net.py            # GCBF Network h(x)  — scalar CBF per agent
│   └── policy_net.py          # Policy Network π(x) — action per agent
└── utils/
    └── graph.py               # Lightweight GraphsTuple (no torch_geometric)
test_step1.py                  # Smoke test for Step 1
```

## Running on Google Colab

Paste the following into the **first cell** of your Colab notebook:

```python
# ── Setup: clone repo & install deps ──
!git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git /content/gcbfplus
%cd /content/gcbfplus
!pip install -q -r requirements.txt

# ── Run the smoke tests ──
!python test_step1.py
```

> **Note**: Replace `<YOUR_USERNAME>/<YOUR_REPO>` with your actual GitHub repo URL.

## Architecture (Table I)

| Network | Encoder ψ₁ | Attention ψ₂ | Decoder ψ₄ |
|--------|-----------|-------------|-----------|
| GCBF h(x) | Input → 256 → 256 → 128 | 128 → 128 → 128 → 1 | 128 → 256 → 256 → **1** |
| Policy π(x) | Input → 256 → 256 → 128 | 128 → 128 → 128 → 1 | 128 → 256 → 256 → **action_dim** |

# cWGAN-GP for Load & Price (USEP) Scenarios

This minimal project trains a **conditional WGAN-GP** to generate **joint sequences** of
`[USEP (price), LOAD]` conditioned on time features and optional exogenous covariates (`PV`, `RP`).

**Data source used here:** your uploaded `/mnt/data/Real_usep_load_pv_rp (2).csv`
with columns: `DATE (half-hourly)`, `PERIOD`, `USEP`, `LOAD`, `PV`, `RP`.

## Quickstart (on your GPU server)

```bash
# 0) (Once) Create/activate env and install deps
conda create -n gan-energy python=3.11 -y
conda activate gan-energy
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pandas numpy scikit-learn matplotlib tqdm

# 1) Put data in data/ (we reference the already-uploaded absolute path by default)
mkdir -p data

# 2) Train (T=48 half-hourly steps = 1 day)
python train.py --data_path "/mnt/data/Real_usep_load_pv_rp (2).csv" --seq_len 48 --epochs 40 --batch_size 64

# 3) Sample 100 scenarios for a given calendar week (48*7 points)
python sample.py --checkpoint "checkpoints/best.pt" --seq_len 48 --n_days 7 --n_samples 100

# 4) Compute import cost quantiles if you have PV (from file) or assume PV=0
python cost_from_scenarios.py --scenarios "outputs/samples_pt.pkl" --pv_source "file" --pv_file "/mnt/data/Real_usep_load_pv_rp (2).csv"
```

## Notes

- The model is **conditional** on: hour-of-day, day-of-week, month, and (optionally) PV & RP.
- Targets: 2 channels `[USEP, LOAD]` generated together, so their co-movements are preserved.
- Loss: WGAN-GP with 1D Conv generator/critic for stability and speed.
- Use the **mean/quantiles** of samples for point forecasts or risk analysis (P50, P90, etc.).
# cWGAN-GP for Load & Price (USEP) Scenarios

This project implements a **conditional WGAN-GP** to generate **joint sequences** of  
**[USEP (Uniform Singapore Energy Price), LOAD]** conditioned on calendar features and optional exogenous covariates (**PV, RP**).  
It also provides tools to **sample scenarios** and **compute import-cost distributions** (P10/P50/P90).

---

## 📊 Problem Statement

- **Targets**  
  - `USEP` (SGD/MWh)  
  - `LOAD` (MW)  

- **Conditioning variables**  
  - Hour-of-day (sin/cos)  
  - Day-of-week (sin/cos)  
  - Month (sin/cos)  
  - Optional exogenous: `PV`, `RP`  

- **Architecture**  
  - 1D Conv **Generator** & **Critic**  
  - Loss: **WGAN-GP**  

- **Outputs**  
  - Shape: `[n_samples, n_days, 2, seq_len]`  
  - Example: `100 samples × 7 days × 2 targets × 48 timesteps`  

- **Cost formula**  
  ```text
  ImportCost = Σ_t USEP[t] * max(0, LOAD[t] – PV[t])
  ```

---

## 🗂 Data

We used a CSV with the following columns:

- `DATE` (half-hourly timestamps)  
- `PERIOD`  
- `USEP`  
- `LOAD`  
- `PV` (photovoltaic generation)  
- `RP` (reserve/regulation price)  

**Paths:**
- Local (Windows):
  ```
  C:\Users\Dell\Downloads\Real_usep_load_pv_rp.csv
  C:\Users\Dell\Downloads\Predicted_usep_load_pv_rp.csv
  ```
- Server (DGX):
  ```
  ~/Real_usep_load_pv_rp.csv
  ~/Predicted_usep_load_pv_rp.csv
  ```
- Project folder:
  ```
  ~/gan_load_price
  ```

---

## 🚀 Getting Started

### 0. Log in to the server
```bash
ssh dgx1570@172.16.203.101
```

### 1. Copy CSVs from Windows to server
> Run these on **Windows**, not on the server.

```powershell
scp "C:/Users/Dell/Downloads/Real_usep_load_pv_rp.csv"      dgx1570@172.16.203.101:~/
scp "C:/Users/Dell/Downloads/Predicted_usep_load_pv_rp.csv" dgx1570@172.16.203.101:~/
```

**Pitfall**: Running `scp "C:/..."` on the server fails (`ssh: Could not resolve hostname c:`).  
Always run from Windows.

---

## ⚙️ Environment Setup

We faced issues with **CUDA mismatches** and **ITTAPI**. Here’s what worked:

### Option A — Conda GPU (cu118)
```bash
conda create -y -n gan-energy-clean python=3.11
conda activate gan-energy-clean
conda install -y pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Option B — Pip GPU (cu118, recommended fix for ITTAPI)
```bash
conda remove -y pytorch pytorch-cuda torchvision torchaudio
pip uninstall -y torch torchvision torchaudio

pip install --upgrade --force-reinstall --no-cache-dir   torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Option C — CPU-only (slower)
```bash
conda create -y -n gan-energy-cpu python=3.11
conda activate gan-energy-cpu
conda install -y pytorch cpuonly -c pytorch
```

**Loader cleanup** (helpful in both cases):
```bash
unset LD_PRELOAD
export LD_LIBRARY_PATH=""
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 📂 Project Structure

```
gan_load_price/
├── train.py
├── sample.py
├── cost_from_scenarios.py
├── model.py
├── utils.py
├── checkpoints/        # best.pt
└── outputs/            # samples_pt.pkl, costs.csv, summary.json
```

---

## 🏋️ Training

Train for 40 epochs (already done in our case):

```bash
conda activate gan-energy-clean
cd ~/gan_load_price

python train.py   --data_path ~/Real_usep_load_pv_rp.csv   --seq_len 48 --epochs 40 --batch_size 64
```

Checkpoints are saved to `checkpoints/best.pt`.

**Run long jobs with `screen`:**
```bash
screen -dmS gantrain bash -lc '   source ~/miniconda3/etc/profile.d/conda.sh &&   conda activate gan-energy-clean &&   cd ~/gan_load_price &&   python -u train.py --data_path ~/Real_usep_load_pv_rp.csv --seq_len 48 --epochs 40 --batch_size 64   |& tee -a train.log '
```

---

## 🔮 Sampling Scenarios

**Important Fix:**  
- Generator input channels must be `z_dim + cond_ch` (not `cond_ch * seq_len`).  
- Condition built from flat covariates → `[n_days, F, T]`.  

```bash
python sample.py   --checkpoint checkpoints/best.pt   --data_path ~/Real_usep_load_pv_rp.csv   --seq_len 48 --n_days 7 --n_samples 100
```

Expected output:
```
Saved samples to outputs/samples_pt.pkl
```

Check:
```bash
python - <<'PY'
import pickle
d = pickle.load(open("outputs/samples_pt.pkl","rb"))
print("shape:", d["samples"].shape)  # (100,7,2,48)
PY
```

---

## 💰 Cost Computation

We extended `cost_from_scenarios.py` to output CSV + JSON.

```bash
python cost_from_scenarios.py   --scenarios outputs/samples_pt.pkl   --pv_source file   --pv_file ~/Real_usep_load_pv_rp.csv   --write_csv --csv_path outputs/costs.csv --per_day   --write_summary --summary_path outputs/cost_summary.json
```

Console:
```
Total import cost over window | P10, P50, P90: [...]
Wrote per-scenario costs to outputs/costs.csv
Wrote summary to outputs/cost_summary.json
```

---

## 📉 Example Results

From our last run:
```
Total import cost over window | P10, P50, P90:
[1,398,904.64   1,517,290.62   1,620,359.93]
```

---

## ⬇️ Downloading Results

From Windows:
```powershell
scp dgx1570@172.16.203.101:~/gan_load_price/outputs/samples_pt.pkl "C:/Users/Dell/Downloads/"
scp dgx1570@172.16.203.101:~/gan_load_price/outputs/costs.csv "C:/Users/Dell/Downloads/"
scp dgx1570@172.16.203.101:~/gan_load_price/outputs/cost_summary.json "C:/Users/Dell/Downloads/"
scp dgx1570@172.16.203.101:~/gan_load_price/checkpoints/best.pt "C:/Users/Dell/Downloads/"
```

Or zip everything on server:
```bash
cd ~/gan_load_price
zip -r outputs_bundle.zip outputs/ checkpoints/best.pt
```

Then download:
```powershell
scp dgx1570@172.16.203.101:~/gan_load_price/outputs_bundle.zip "C:/Users/Dell/Downloads/"
```

---

## ⚠️ Pitfalls & Fixes

- **CUDA mismatch / ITTAPI errors**  
  → Use pip cu118 wheels; clean env vars with `unset LD_PRELOAD`.

- **State dict size mismatch (72 vs 448)**  
  → Ensure Generator input = `z_dim + cond_ch`. Build `cond` as `[n_days, F, T]`.

- **Tilde quoting**  
  → Do not quote: `--data_path ~/file.csv` (not `"~/file.csv"`).

- **SCP from wrong place**  
  → Run `scp` on Windows, not inside the server. Extract zips first.

---

## 🎯 Golden Path (no retraining)

```bash
conda activate gan-energy-clean
cd ~/gan_load_price

# 1. Sample
python sample.py   --checkpoint checkpoints/best.pt   --data_path ~/Real_usep_load_pv_rp.csv   --seq_len 48 --n_days 7 --n_samples 100

# 2. Cost analysis
python cost_from_scenarios.py   --scenarios outputs/samples_pt.pkl   --pv_source file   --pv_file ~/Real_usep_load_pv_rp.csv   --write_csv --csv_path outputs/costs.csv --per_day   --write_summary --summary_path outputs/cost_summary.json
```

---

## 📌 License & Credits

Minimal framework adapted for Singapore USEP data scenario generation and cost analysis.  
Use responsibly and validate against your metrics.

cWGAN-GP for Load & Price (USEP) Scenarios

This project trains a conditional WGAN-GP to generate joint sequences of [USEP (price), LOAD] conditioned on calendar features and optional exogenous covariates (PV, RP). It also lets you sample scenarios from a trained model and compute import-cost distributions (P10/P50/P90).

Targets: USEP (SGD/MWh), LOAD (MW)

Conditioning: hour-of-day (sin/cos), day-of-week (sin/cos), month (sin/cos), + optional PV, RP (all standardized)

Loss/arch: 1D-Conv Generator/Critic with WGAN-GP

Output: samples shaped [S, D, 2, T] where S = n_samples, D = n_days, T = seq_len (e.g., 48 per day)

Cost formula: Σ_t USEP[t] * max(0, LOAD[t] − PV[t])

A. Data

We used a CSV with columns:

DATE (half-hourly timestamps), PERIOD, USEP, LOAD, PV, RP

Example paths:

Windows local:

C:\Users\Dell\Downloads\Real_usep_load_pv_rp.csv
C:\Users\Dell\Downloads\Predicted_usep_load_pv_rp.csv


Server (home = ~):

~/Real_usep_load_pv_rp.csv
~/Predicted_usep_load_pv_rp.csv


Project folder: ~/gan_load_price

B. Logging in to the server & copying files
1) SSH into the server

From Windows (PowerShell/CMD):

ssh dgx1570@172.16.203.101

2) Copy files from Windows to server (run on Windows)

Use forward slashes in paths; do not run these while you’re already SSH’d into the server.

scp "C:/Users/Dell/Downloads/Real_usep_load_pv_rp.csv"      dgx1570@172.16.203.101:~/
scp "C:/Users/Dell/Downloads/Predicted_usep_load_pv_rp.csv" dgx1570@172.16.203.101:~/


Pitfall:

Running scp "C:/..." on the server fails: ssh: Could not resolve hostname c:
Always run scp on your Windows machine.

C:\Users\...\gan_load_price.zip\sample.py isn’t a real path — extract the zip first.

C. Environment setup (CUDA / PyTorch gotchas & fixes)

We saw two common issues:

cuSparse / libnvJitLink mismatch (Torch cu12 wheels vs installed driver)

libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent (missing libittnotify.so from ittapi)

Choose ONE working path

Option A — Conda (GPU cu118)

conda create -y -n gan-energy-clean python=3.11
conda activate gan-energy-clean
conda install -y pytorch pytorch-cuda=11.8 -c pytorch -c nvidia


Option B — Pip cu118 wheels (this avoided the ITTAPI issue for us)

# remove any conda torch first (don't mix conda/pip builds)
conda remove -y pytorch pytorch-cuda torchvision torchaudio
pip uninstall -y torch torchvision torchaudio

# install official cu118 wheels
pip install --upgrade --force-reinstall --no-cache-dir \
  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


Option C — CPU-only (quick unblock)

conda create -y -n gan-energy-cpu python=3.11
conda activate gan-energy-cpu
conda install -y pytorch cpuonly -c pytorch


Loader cleanup (useful in both cases)

unset LD_PRELOAD
export LD_LIBRARY_PATH=""
python -c "import torch; print('torch',torch.__version__,'cuda?',torch.cuda.is_available())"

D. Project layout
gan_load_price/
├── train.py
├── sample.py
├── cost_from_scenarios.py
├── model.py
├── utils.py
├── checkpoints/           # best.pt saved here
└── outputs/               # samples_pt.pkl, costs.csv, cost_summary.json

E. Training (done once — you already ran 40 epochs)
conda activate gan-energy-clean
cd ~/gan_load_price

python -u train.py \
  --data_path ~/Real_usep_load_pv_rp.csv \
  --seq_len 48 --epochs 40 --batch_size 64 | tee -a train.log


Checkpoints saved to checkpoints/best.pt with metadata: seq_len, z_dim, hidden, and scalers.

Training log example you saw:

Epoch 034 ... VA-MAE 0.7218
...
Epoch 040 ... VA-MAE 0.7373


Run long jobs safely with screen

screen -dmS gantrain bash -lc ' \
  source ~/miniconda3/etc/profile.d/conda.sh && \
  conda activate gan-energy-clean && \
  cd ~/gan_load_price && \
  python -u train.py --data_path ~/Real_usep_load_pv_rp.csv --seq_len 48 --epochs 40 --batch_size 64 \
  |& tee -a train.log \
'
screen -ls
screen -r gantrain   # attach; detach with Ctrl+A, D
screen -S gantrain -X quit


Pitfall: Do not quote the tilde ("~/Real...") when passing --data_path. Tilde won’t expand inside quotes.

F. Sampling (no retraining)

Critical architecture note:
Your checkpoint uses non-stacked conditioning. The Generator in-channels must be z_dim + cond_ch (e.g., 64 + 8 = 72).
If you time-stack F*T, you’ll get 448 (64 + 8*48) and state_dict loading will fail.

In model.py (correct):

self.init = nn.Conv1d(z_dim + cond_ch, hidden, 3, 1, 1)


In sample.py (correct):

Build cond from flat covariate series, take the last n_days * seq_len rows, and reshape to [n_days, F, T].

Do not flatten across time.

Run sampling

conda activate gan-energy-clean
cd ~/gan_load_price

python sample.py \
  --checkpoint checkpoints/best.pt \
  --data_path ~/Real_usep_load_pv_rp.csv \
  --seq_len 48 --n_days 7 --n_samples 100


Expected:

Saved samples to outputs/samples_pt.pkl


The file is small (a few hundred KB for 100×7×2×48 float32).

Sanity checks

ls -lh outputs/samples_pt.pkl

python - <<'PY'
import pickle
d = pickle.load(open("outputs/samples_pt.pkl","rb"))
print("keys:", d.keys())
print("shape:", d["samples"].shape)  # expect (100,7,2,48)
print("scaled:", d.get("scaled"))    # False if inverse-transformed to real units
PY

G. Cost computation (CSV + JSON outputs)

We added CSV/JSON outputs to cost_from_scenarios.py. Example:

python cost_from_scenarios.py \
  --scenarios outputs/samples_pt.pkl \
  --pv_source file \
  --pv_file ~/Real_usep_load_pv_rp.csv \
  --write_csv --csv_path outputs/costs.csv --per_day \
  --write_summary --summary_path outputs/cost_summary.json


Console prints:

Total import cost over window | P10, P50, P90: [ ... ]
Wrote per-scenario costs to outputs/costs.csv
Wrote summary to outputs/cost_summary.json


outputs/costs.csv → per-scenario totals + optional per-day columns

outputs/cost_summary.json → P10/P50/P90 and shape metadata

Variants

No PV baseline: --pv_source zero

Use predicted PV: --pv_file ~/Predicted_usep_load_pv_rp.csv (must align in cadence & length)

H. Common pitfalls & fixes (we hit these!)

CUDA/ITTAPI loader errors

Symptom: libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent

Fix: use pip cu118 wheels (Option B), then:

unset LD_PRELOAD
export LD_LIBRARY_PATH=""


cuSparse / libnvJitLink mismatch

Symptom: Torch cu12 wheels with older driver

Fix: install cu118 builds (Option A or B above).

State dict size mismatch: 72 vs 448

Symptom:

size mismatch for init.weight: ... ckpt (128, 72, 3) vs model (128, 448, 3)


Cause: time-stacked conditioning in the Generator (or building cond incorrectly)

Fix:

Ensure model.py uses z_dim + cond_ch (not cond_ch * seq_len)

In sample.py, slice from flat covariates, reshape to [n_days, F, T], no flatten

Quick check:

python - <<'PY'
import torch
ckpt = torch.load("checkpoints/best.pt", map_location="cpu")
print("ckpt init.weight:", tuple(ckpt["gen"]["init.weight"].shape))  # (128,72,3)
print("z_dim:", ckpt["z_dim"], "cond_ch expected:", ckpt["gen"]["init.weight"].shape[1]-ckpt["z_dim"])
PY


Tilde expansion & quoting

--data_path "~/file.csv" → tilde won’t expand. Use --data_path ~/file.csv or $HOME/file.csv.

SCP from wrong place / zipped paths

Running scp "C:/..." on the server → ssh: Could not resolve hostname c:

Extract zips; use real paths (e.g., C:/Users/Dell/Downloads/gan_load_price/sample.py) and run scp on Windows.

Sampling units

Make sure sample.py inverse-transforms outputs using checkpoint y_scaler_mean/scale so USEP/LOAD are in real units.

I. Golden path (no retraining)
# 1) Activate env
conda activate gan-energy-clean
cd ~/gan_load_price

# 2) Sampling (real units; quick)
python sample.py \
  --checkpoint checkpoints/best.pt \
  --data_path ~/Real_usep_load_pv_rp.csv \
  --seq_len 48 --n_days 7 --n_samples 100

# 3) Cost distribution + artifacts
python cost_from_scenarios.py \
  --scenarios outputs/samples_pt.pkl \
  --pv_source file \
  --pv_file ~/Real_usep_load_pv_rp.csv \
  --write_csv --csv_path outputs/costs.csv --per_day \
  --write_summary --summary_path outputs/cost_summary.json

J. Downloading results to your PC

From Windows:

# Scenarios
scp dgx1570@172.16.203.101:~/gan_load_price/outputs/samples_pt.pkl "C:/Users/Dell/Downloads/"

# Costs + summary
scp dgx1570@172.16.203.101:~/gan_load_price/outputs/costs.csv "C:/Users/Dell/Downloads/"
scp dgx1570@172.16.203.101:~/gan_load_price/outputs/cost_summary.json "C:/Users/Dell/Downloads/"

# (Optional) Checkpoint
scp dgx1570@172.16.203.101:~/gan_load_price/checkpoints/best.pt "C:/Users/Dell/Downloads/"


Bundle & grab in one go

# on server
cd ~/gan_load_price
zip -r outputs_bundle.zip outputs/ checkpoints/best.pt

# on Windows
scp dgx1570@172.16.203.101:~/gan_load_price/outputs_bundle.zip "C:/Users/Dell/Downloads/"


GUI option: WinSCP (SFTP) → Host 172.16.203.101, user dgx1570.

K. Performance notes

Sampling is fast: n_samples × n_days forward passes of a small 1D-Conv net.
Typical runtime: a few seconds on CPU; ~1–3 s on GPU.

File sizes stay small (hundreds of KB to a few MB).

L. Troubleshooting checklist

torch.cuda.is_available() is False? → Sampling still fine on CPU; training slower.

KeyError: "y_scaler_mean" when sampling? → You’re using an older checkpoint; re-train or skip inverse-transform (costs would be incorrect).

PV alignment errors? → Ensure PV column exists; last n_days*seq_len rows match the sampled window.

“No such file or directory” → Check paths, avoid quoting ~.

M. Credits & License

This minimal framework is adapted for quick, reproducible scenario generation and cost analysis on half-hourly Singapore USEP data with optional PV/RP covariates. Use responsibly and validate against your business metrics./critic for stability and speed.
- Use the **mean/quantiles** of samples for point forecasts or risk analysis (P50, P90, etc.).

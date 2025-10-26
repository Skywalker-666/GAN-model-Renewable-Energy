#!/usr/bin/env python3
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalize_time_axis(T):
    xs = np.arange(T)
    labels = []
    for t in xs:
        h = (t // 2) % 24
        m = (t % 2) * 30
        labels.append(f"{h:02d}:{m:02d}")
    return xs, labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)     # outputs_run1/samples.npz
    ap.add_argument("--real", required=True)        # real_data.csv
    ap.add_argument("--date", required=True)        # e.g., 2021-08-05
    ap.add_argument("--n", type=int, default=10)    # number of scenarios
    ap.add_argument("--outdir", default="figs_10")
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load scenarios
    Z = np.load(args.samples, allow_pickle=True)
    dates = pd.to_datetime(Z["dates"]).strftime("%Y-%m-%d")
    usep_samps = Z["usep_samples"]  # [D,S,T]
    load_samps = Z["load_samples"]  # [D,S,T]
    D, S, T = usep_samps.shape

    # Find requested day
    if args.date not in dates:
        raise SystemExit(f"Date {args.date} not found in samples. Available span: {dates.min()} .. {dates.max()}")
    d_idx = np.where(dates == args.date)[0][0]

    # Cap scenarios to min(S, n)
    K = min(S, args.n)

    # Load real data for that day
    df = pd.read_csv(args.real)
    df.columns = [c.strip().upper() for c in df.columns]
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.normalize()
    day = df[df["DATE"] == pd.to_datetime(args.date)]
    if len(day) < T:
        raise SystemExit(f"Real data for {args.date} has only {len(day)} rows, need {T}.")
    usep_real = day["USEP"].values[:T].astype(float)
    load_real = day["LOAD"].values[:T].astype(float)

    xticks, xlabels = normalize_time_axis(T)

    # -------- USEP plot
    fig, ax = plt.subplots(figsize=(14,4))
    for s in range(K):
        ax.plot(xticks, usep_samps[d_idx, s], alpha=0.35, linewidth=1)
    ax.plot(xticks, usep_real, color="k", lw=3, label="Actual USEP")
    ax.set_title(f"USEP — {args.date} | {K} scenarios vs Actual")
    ax.set_ylabel("USEP (SGD/MWh)")
    ax.set_xticks(xticks[::4]); ax.set_xticklabels(xlabels[::4])
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, f"usep_{args.date}_10scenarios.png"), dpi=args.dpi)
    plt.close(fig)

    # -------- LOAD plot
    fig, ax = plt.subplots(figsize=(14,4))
    for s in range(K):
        ax.plot(xticks, load_samps[d_idx, s], alpha=0.35, linewidth=1)
    ax.plot(xticks, load_real, color="k", lw=3, label="Actual LOAD")
    ax.set_title(f"LOAD — {args.date} | {K} scenarios vs Actual")
    ax.set_ylabel("LOAD (MW)")
    ax.set_xticks(xticks[::4]); ax.set_xticklabels(xlabels[::4])
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, f"load_{args.date}_10scenarios.png"), dpi=args.dpi)
    plt.close(fig)

    print(f"[OK] Wrote {os.path.join(args.outdir, f'usep_{args.date}_10scenarios.png')}")
    print(f"[OK] Wrote {os.path.join(args.outdir, f'load_{args.date}_10scenarios.png')}")

if __name__ == "__main__":
    main()

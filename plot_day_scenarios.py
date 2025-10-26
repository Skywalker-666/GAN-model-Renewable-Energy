#!/usr/bin/env python3
"""
Overlay ALL scenarios vs. Actual for a given day range.
Pure visualization (no selection, no Top-K).

Inputs
------
--samples : same format as select_and_plot_topK.py
--real    : CSV with DATE, USEP, LOAD (PERIOD optional)
--start   : inclusive date (YYYY-MM-DD)
--end     : inclusive date (YYYY-MM-DD)
--outdir  : directory for figures
--dpi     : figure dpi
--thin    : optional int to thin scenario lines (plot every 'thin'-th scenario)

Produces
--------
One PNG per day for USEP and LOAD (two files/day), with all scenario lines and actual.
"""

import os, argparse, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (14, 4)
plt.rcParams["axes.grid"] = True

def load_samples(path):
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        return {
            "dates": z["dates"],
            "time": z["time"],
            "usep_samples": z["usep_samples"],
            "load_samples": z["load_samples"],
        }
    with open(path, "rb") as f:
        z = pickle.load(f)
        return z

def normalize_time_axis(T):
    ticks = np.arange(T)
    labels = []
    for t in ticks:
        h = (t // 2) % 24
        m = (t % 2) * 30
        labels.append(f"{h:02d}:{m:02d}")
    return ticks, labels

def read_real_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().upper() for c in df.columns]
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"])
    return df

def extract_day_series(df, the_date, T):
    day = df[df["DATE"] == pd.to_datetime(the_date)].sort_index()
    if "USEP" in day and "LOAD" in day and len(day) >= T:
        return day["USEP"].values[:T], day["LOAD"].values[:T]
    u = float(day["USEP"].iloc[0]); l = float(day["LOAD"].iloc[0])
    return np.repeat(u, T), np.repeat(l, T)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--real", required=True)
    ap.add_argument("--start", required=False)
    ap.add_argument("--end", required=False)
    ap.add_argument("--outdir", default="figs_scenarios")
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--thin", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    S = load_samples(args.samples)
    dates = pd.to_datetime(S["dates"])
    usep_samps = np.asarray(S["usep_samples"])   # [D,S,T]
    load_samps = np.asarray(S["load_samples"])   # [D,S,T]
    D, S_, T = usep_samps.shape
    xticks, xlabels = normalize_time_axis(T)
    df_real = read_real_csv(args.real)

    # filter date range
    if args.start:
        mask = dates >= pd.to_datetime(args.start)
        dates = dates[mask]; usep_samps = usep_samps[mask]; load_samps = load_samps[mask]
    if args.end:
        mask = dates <= pd.to_datetime(args.end)
        dates = dates[mask]; usep_samps = usep_samps[mask]; load_samps = load_samps[mask]

    for d_idx, day in enumerate(dates):
        usep_day = usep_samps[d_idx]   # [S,T]
        load_day = load_samps[d_idx]   # [S,T]
        usep_real, load_real = extract_day_series(df_real, day, T)

        # --- USEP
        fig, ax = plt.subplots()
        # plot scenarios (thinned)
        for s in range(0, usep_day.shape[0], args.thin):
            ax.plot(xticks, usep_day[s], alpha=0.3, linewidth=1)
        ax.plot(xticks, usep_real, color="k", lw=3, label="Actual USEP")
        ax.set_title(f"USEP — Actual vs. All Scenarios | {day.date()}")
        ax.set_ylabel("USEP (SGD/MWh)")
        ax.set_xticks(xticks[::4]); ax.set_xticklabels(xlabels[::4])
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, f"usep_all_{day.date()}.png"), dpi=args.dpi)
        plt.close(fig)

        # --- LOAD
        fig, ax = plt.subplots()
        for s in range(0, load_day.shape[0], args.thin):
            ax.plot(xticks, load_day[s], alpha=0.3, linewidth=1)
        ax.plot(xticks, load_real, color="k", lw=3, label="Actual LOAD")
        ax.set_title(f"LOAD — Actual vs. All Scenarios | {day.date()}")
        ax.set_ylabel("LOAD (MW)")
        ax.set_xticks(xticks[::4]); ax.set_xticklabels(xlabels[::4])
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, f"load_all_{day.date()}.png"), dpi=args.dpi)
        plt.close(fig)

    print(f"[OK] Plots written to {args.outdir}")

if __name__ == "__main__":
    main()

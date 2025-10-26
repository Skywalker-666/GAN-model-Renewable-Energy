#!/usr/bin/env python3
"""
Honest evaluation & plotting for USEP/LOAD scenario forecasts.
- No post-hoc Top-K selection (no leakage).
- Plots median and P10–P90 bands vs. Actual.
- Reports Energy Score, Coverage, and RMSE(median).

Inputs
------
--samples  : path to a npz/pickle with arrays:
             'dates' [D], 'time' [T], 'usep_samples' [D,S,T], 'load_samples' [D,S,T]
--real     : CSV with columns: DATE, PERIOD(optional), USEP, LOAD
--outdir   : directory for figures
--dpi      : figure dpi (default 140)

Notes
-----
- Channels order in samples: usep first, load second (kept separate here for clarity).
- DATE should parse to pandas datetime; PERIOD optional (half-hourly index).
"""

import os, argparse, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

plt.rcParams["figure.figsize"] = (14, 4)
plt.rcParams["axes.grid"] = True

def energy_score(samples, y):
    # samples: [S,T], y: [T]
    s = np.asarray(samples)
    y = np.asarray(y)
    term1 = np.mean(np.sqrt(np.sum((s - y[None, :]) ** 2, axis=1)))
    diffs = s[:, None, :] - s[None, :, :]
    term2 = 0.5 * np.mean(np.sqrt(np.sum(diffs ** 2, axis=2)))
    return term1 - term2

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

def normalize_time_axis(time_like):
    """Make a pretty x-axis (e.g., 48 half-hours -> '00:00'..)."""
    if isinstance(time_like, np.ndarray) and np.issubdtype(time_like.dtype, np.number):
        T = len(time_like)
        ticks = np.arange(T)
        labels = []
        for t in ticks:
            h = (t // 2) % 24
            m = (t % 2) * 30
            labels.append(f"{h:02d}:{m:02d}")
        return ticks, labels
    return np.arange(len(time_like)), list(map(str, time_like))

def read_real_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().upper() for c in df.columns]
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"])
    return df

def extract_day_series(df, the_date, T):
    # Expect possibly multiple PERIOD rows per DATE
    day = df[df["DATE"] == pd.to_datetime(the_date)].sort_index()
    if "USEP" in day and "LOAD" in day and len(day) >= T:
        usep = day["USEP"].values[:T]
        load = day["LOAD"].values[:T]
        return usep, load
    # fallback: single-row daily data (not typical), tile
    u = float(day["USEP"].iloc[0]); l = float(day["LOAD"].iloc[0])
    return np.repeat(u, T), np.repeat(l, T)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--real", required=True)
    ap.add_argument("--outdir", default="figs_eval")
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    S = load_samples(args.samples)
    dates = pd.to_datetime(S["dates"])
    time_axis = S["time"]
    usep_samps = np.asarray(S["usep_samples"])   # [D,S,T]
    load_samps = np.asarray(S["load_samples"])   # [D,S,T]
    D, S_, T = usep_samps.shape

    df_real = read_real_csv(args.real)
    xticks, xlabels = normalize_time_axis(time_axis)

    # aggregate metrics
    rows = []
    for d_idx, day in enumerate(dates):
        usep_day = usep_samps[d_idx]        # [S,T]
        load_day = load_samps[d_idx]        # [S,T]
        usep_p10 = np.percentile(usep_day, 10, axis=0)
        usep_p50 = np.percentile(usep_day, 50, axis=0)
        usep_p90 = np.percentile(usep_day, 90, axis=0)
        load_p10 = np.percentile(load_day, 10, axis=0)
        load_p50 = np.percentile(load_day, 50, axis=0)
        load_p90 = np.percentile(load_day, 90, axis=0)

        usep_real, load_real = extract_day_series(df_real, day, T)

        # Metrics
        es_usep = energy_score(usep_day, usep_real)
        es_load = energy_score(load_day, load_real)
        cov_usep = np.mean((usep_p10 <= usep_real) & (usep_real <= usep_p90))
        cov_load = np.mean((load_p10 <= load_real) & (load_real <= load_p90))
        rmse_usep = np.sqrt(np.mean((usep_p50 - usep_real) ** 2))
        rmse_load = np.sqrt(np.mean((load_p50 - load_real) ** 2))

        rows.append({
            "date": day.date(),
            "ES_USEP": es_usep, "COV_USEP": cov_usep, "RMSE_MED_USEP": rmse_usep,
            "ES_LOAD": es_load, "COV_LOAD": cov_load, "RMSE_MED_LOAD": rmse_load
        })

        # ---- Plot USEP
        fig, ax = plt.subplots()
        ax.fill_between(xticks, usep_p10, usep_p90, alpha=0.25, label="P10–P90")
        ax.plot(xticks, usep_p50, lw=2, label="Median")
        ax.plot(xticks, usep_real, lw=3, color="k", label="Actual")
        ax.set_title(f"USEP — Scenarios vs Actual | {day.date()}")
        ax.set_ylabel("USEP (SGD/MWh)")
        ax.set_xticks(xticks[::4]); ax.set_xticklabels(xlabels[::4], rotation=0)
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, f"usep_{day.date()}.png"), dpi=args.dpi)
        plt.close(fig)

        # ---- Plot LOAD
        fig, ax = plt.subplots()
        ax.fill_between(xticks, load_p10, load_p90, alpha=0.25, label="P10–P90")
        ax.plot(xticks, load_p50, lw=2, label="Median")
        ax.plot(xticks, load_real, lw=3, color="k", label="Actual")
        ax.set_title(f"LOAD — Scenarios vs Actual | {day.date()}")
        ax.set_ylabel("LOAD (MW)")
        ax.set_xticks(xticks[::4]); ax.set_xticklabels(xlabels[::4], rotation=0)
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, f"load_{day.date()}.png"), dpi=args.dpi)
        plt.close(fig)

    # Save metrics CSV
    pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "metrics_summary.csv"), index=False)
    print(f"[OK] Wrote figures and metrics to {args.outdir}")

if __name__ == "__main__":
    main()

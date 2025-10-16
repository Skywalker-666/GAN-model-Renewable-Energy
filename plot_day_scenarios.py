import argparse, os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, HourLocator

def load_last_days(df, date_col, days, seq_len):
    """Return the last `days` calendar days as a (dates_list, ts_index, day2slice) mapping."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    # extract unique days (date part only) and take the last `days`
    df["__day__"] = df[date_col].dt.date
    uniq_days = df["__day__"].drop_duplicates().to_list()
    if len(uniq_days) < days:
        raise ValueError(f"CSV has only {len(uniq_days)} unique days; need {days}.")
    selected_days = uniq_days[-days:]
    # Build a day->index mapping and slices for 48 steps each
    day_slices = {}
    ts_index = {}
    for d in selected_days:
        mask = df["__day__"] == d
        day_df = df.loc[mask].sort_values(date_col)
        if len(day_df) < seq_len:
            raise ValueError(f"Day {d} has only {len(day_df)} rows; need {seq_len}.")
        # take the last seq_len points of that day in case of more rows
        day_df = day_df.iloc[-seq_len:]
        ts_index[d] = day_df[date_col].values
        day_slices[d] = day_df.index
    return selected_days, ts_index, day_slices

def main(args):
    os.makedirs("outputs/figs", exist_ok=True)

    # --- Load scenarios ---
    with open(args.scenarios, "rb") as f:
        data = pickle.load(f)
    samples = data["samples"]                      # [S, D, 2, T]
    S, D, C, T = samples.shape
    assert C == 2, "Expected channels [USEP, LOAD]"
    USEP = samples[:, :, 0, :]                     # [S, D, T]
    LOAD = samples[:, :, 1, :]                     # [S, D, T]

    # --- Load real CSV and align to last D days (or a specific date) ---
    df = pd.read_csv(args.csv_path, parse_dates=[args.date_col])
    days, ts_index, day_slices = load_last_days(df, args.date_col, D, T)

    # If user provided a specific date, map it to the day index among the last D days
    if args.date:
        target = pd.to_datetime(args.date).date()
        if target not in days:
            raise SystemExit(f"--date {args.date} not found within the last {D} day(s): {days}")
        day_idx = days.index(target)
    else:
        day_idx = D - 1  # by default, use the most recent day among the last D

    # Build actual series (vectors of len T) for the chosen day
    the_day = days[day_idx]
    rows = day_slices[the_day]
    # ensure correct order (in case index is not sorted)
    day_df = df.loc[rows].sort_values(args.date_col)
    ts = pd.to_datetime(day_df[args.date_col].values)
    actual_price = day_df[args.usep_col].values
    actual_load  = day_df[args.load_col].values

    # Sanity
    if len(actual_price) != T or len(actual_load) != T:
        raise ValueError(f"Day {the_day} has len {len(actual_price)}, expected {T}.")

    # --- PLOT HELPERS ---
    def format_time_axis(ax):
        ax.xaxis.set_major_locator(HourLocator(interval=2))
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d\n%H:%M"))
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # === A) OVERLAY: ALL SCENARIOS vs ACTUAL (PRICE) ===
    fig, ax = plt.subplots(figsize=(14,6))
    for s in range(S):
        ax.plot(ts, USEP[s, day_idx, :], alpha=0.25)
    ax.plot(ts, actual_price, linewidth=2.5, label="Actual USEP")
    ax.set_title(f"USEP — All {S} scenarios vs Actual | Day {the_day}")
    ax.set_xlabel("Date Time")
    ax.set_ylabel("USEP (SGD/MWh)")
    ax.legend()
    format_time_axis(ax)
    fig.tight_layout()
    out_p_all = f"outputs/figs/price_all_scenarios_day{day_idx}.png"
    fig.savefig(out_p_all, dpi=150)
    plt.close(fig)

    # === B) OVERLAY: ALL SCENARIOS vs ACTUAL (LOAD) ===
    fig, ax = plt.subplots(figsize=(14,6))
    for s in range(S):
        ax.plot(ts, LOAD[s, day_idx, :], alpha=0.25)
    ax.plot(ts, actual_load, linewidth=2.5, label="Actual LOAD")
    ax.set_title(f"LOAD — All {S} scenarios vs Actual | Day {the_day}")
    ax.set_xlabel("Date Time")
    ax.set_ylabel("LOAD (MW)")
    ax.legend()
    format_time_axis(ax)
    fig.tight_layout()
    out_l_all = f"outputs/figs/load_all_scenarios_day{day_idx}.png"
    fig.savefig(out_l_all, dpi=150)
    plt.close(fig)

    # === C) PER-SCENARIO PANELS vs ACTUAL ===
    max_panels = min(args.max_panels, S)

    for s in range(max_panels):
        # Price
        fig, ax = plt.subplots(figsize=(14,6))
        ax.plot(ts, actual_price, linewidth=2.5, label="Actual USEP")
        ax.plot(ts, USEP[s, day_idx, :], linewidth=1.5, label=f"Scenario {s}")
        ax.set_title(f"USEP — Actual vs Scenario {s} | Day {the_day}")
        ax.set_xlabel("Date Time")
        ax.set_ylabel("USEP (SGD/MWh)")
        ax.legend()
        format_time_axis(ax)
        fig.tight_layout()
        fig.savefig(f"outputs/figs/price_vs_scenario_{s}_day{day_idx}.png", dpi=150)
        plt.close(fig)

        # Load
        fig, ax = plt.subplots(figsize=(14,6))
        ax.plot(ts, actual_load, linewidth=2.5, label="Actual LOAD")
        ax.plot(ts, LOAD[s, day_idx, :], linewidth=1.5, label=f"Scenario {s}")
        ax.set_title(f"LOAD — Actual vs Scenario {s} | Day {the_day}")
        ax.set_xlabel("Date Time")
        ax.set_ylabel("LOAD (MW)")
        ax.legend()
        format_time_axis(ax)
        fig.tight_layout()
        fig.savefig(f"outputs/figs/load_vs_scenario_{s}_day{day_idx}.png", dpi=150)
        plt.close(fig)

    print("Saved:")
    print(" ", out_p_all)
    print(" ", out_l_all)
    print(f" Per-scenario panels (first {max_panels}) in outputs/figs/")
    print(" Done.")
    

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", type=str, default="outputs/samples_pt.pkl")
    p.add_argument("--csv_path", type=str, default="~/Real_usep_load_pv_rp.csv")
    p.add_argument("--date", type=str, default=None, help="YYYY-MM-DD (must be within last D days used for sampling)")
    p.add_argument("--usep_col", type=str, default="USEP")
    p.add_argument("--load_col", type=str, default="LOAD")
    p.add_argument("--date_col", type=str, default="DATE")
    p.add_argument("--max_panels", type=int, default=5, help="how many per-scenario panels to save")
    args = p.parse_args()
    # expand ~ in paths
    args.csv_path = os.path.expanduser(args.csv_path)
    args.scenarios = os.path.expanduser(args.scenarios)
    main(args)


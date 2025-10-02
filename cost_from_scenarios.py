import argparse, pickle, numpy as np, pandas as pd, json, os

def main(args):
    # Load scenarios
    with open(args.scenarios, "rb") as f:
        data = pickle.load(f)
    S, D, C, T = data["samples"].shape  # [samples, days, channels(2), T]
    assert C == 2, "Expected 2 target channels: [USEP, LOAD]"
    samples = data["samples"]  # real units if you used the patched sample.py
    # samples[s, d, 0, t] = USEP; samples[s, d, 1, t] = LOAD

    # Get PV for the same calendar block (last D*T rows)
    if args.pv_source == "file":
        df = pd.read_csv(args.pv_file, parse_dates=[args.date_col])
        pv_all = df[args.pv_col].values
        if len(pv_all) < D*T:
            raise ValueError(f"PV file too short: need {D*T} rows, got {len(pv_all)}")
        pv = pv_all[-(D*T):].reshape(D, T)  # [D,T]
    else:
        pv = np.zeros((D, T), dtype=np.float32)

    # Compute import cost per-step, per-day, and total
    USEP = samples[:, :, 0, :]  # [S,D,T]
    LOAD = samples[:, :, 1, :]  # [S,D,T]
    PV   = pv[None, :, :]       # [1,D,T] broadcast to [S,D,T]

    import_energy = np.maximum(0.0, LOAD - PV)     # [S,D,T]
    cost = (USEP * import_energy)                  # [S,D,T]
    per_day = cost.sum(axis=-1)                    # [S,D]
    total   = per_day.sum(axis=-1)                 # [S]

    q10, q50, q90 = np.percentile(total, [10, 50, 90])
    print("Total import cost over window | P10, P50, P90:", [q10, q50, q90])

    # Optional: write detailed CSV
    if args.write_csv:
        os.makedirs(os.path.dirname(args.csv_path) or ".", exist_ok=True)
        out = {"scenario_id": np.arange(S, dtype=int), "total_cost": total}
        if args.per_day:
            for d in range(D):
                out[f"day_{d}_cost"] = per_day[:, d]
        df_out = pd.DataFrame(out)
        df_out.to_csv(args.csv_path, index=False)
        print(f"Wrote per-scenario costs to {args.csv_path}")

    # Optional: write a tiny JSON summary
    if args.write_summary:
        os.makedirs(os.path.dirname(args.summary_path) or ".", exist_ok=True)
        summary = {
            "samples": S, "days": D, "timesteps_per_day": T,
            "p10_total": float(q10), "p50_total": float(q50), "p90_total": float(q90),
            "units": {"USEP": "SGD/MWh", "LOAD": "MW", "PV": "MW"}
        }
        with open(args.summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote summary to {args.summary_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", type=str, required=True)
    p.add_argument("--pv_source", type=str, choices=["file","zero"], default="file")
    p.add_argument("--pv_file", type=str, default="~/Real_usep_load_pv_rp.csv")
    p.add_argument("--pv_col", type=str, default="PV")
    p.add_argument("--date_col", type=str, default="DATE")

    # outputs
    p.add_argument("--write_csv", action="store_true")
    p.add_argument("--csv_path", type=str, default="outputs/costs.csv")
    p.add_argument("--per_day", action="store_true")

    p.add_argument("--write_summary", action="store_true")
    p.add_argument("--summary_path", type=str, default="outputs/cost_summary.json")

    args = p.parse_args()
    main(args)

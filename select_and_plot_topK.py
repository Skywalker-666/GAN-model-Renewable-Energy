import os, argparse, pickle, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, HourLocator

def load_last_days(df, date_col, days, seq_len):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    df["__day__"] = df[date_col].dt.date
    uniq = df["__day__"].drop_duplicates().to_list()
    selected = uniq[-days:]
    day_ts, day_vals = {}, {}
    for d in selected:
        sub = df[df["__day__"]==d].sort_values(date_col).iloc[-seq_len:]
        if len(sub)!=seq_len: continue
        day_ts[d] = pd.to_datetime(sub[date_col].values)
        day_vals[d] = sub
    return selected, day_ts, day_vals

def rmse(a,b): return np.sqrt(np.mean((a-b)**2))
def rank_scenarios(actual, scenarios):
    scores = np.array([rmse(actual, scenarios[s]) for s in range(scenarios.shape[0])])
    idx = np.argsort(scores)
    return idx, scores

def format_time_axis(ax):
    ax.xaxis.set_major_locator(HourLocator(interval=2))
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))

def main(args):
    os.makedirs("outputs/figs", exist_ok=True)
    with open(args.scenarios, "rb") as f: d = pickle.load(f)
    S,D,C,T = d["samples"].shape
    USEP, LOAD = d["samples"][:,:,0,:], d["samples"][:,:,1,:]

    df = pd.read_csv(args.csv_path, parse_dates=[args.date_col])
    days, day_ts, day_vals = load_last_days(df, args.date_col, D, T)

    # pick best-fit day
    best_day, best_score, day_idx = None, float("inf"), None
    for i,dte in enumerate(days):
        act_p, act_l = day_vals[dte][args.usep_col].values, day_vals[dte][args.load_col].values
        best_p = np.min([rmse(act_p, USEP[s,i,:]) for s in range(S)])
        best_l = np.min([rmse(act_l, LOAD[s,i,:]) for s in range(S)])
        score = best_p + best_l
        if score < best_score: best_score, best_day, day_idx = score, dte, i

    ts = day_ts[best_day]
    act_p, act_l = day_vals[best_day][args.usep_col].values, day_vals[best_day][args.load_col].values
    p_order,p_scores = rank_scenarios(act_p, USEP[:,day_idx,:])
    l_order,l_scores = rank_scenarios(act_l, LOAD[:,day_idx,:])
    K = min(args.top_k, S)

    # Save CSV ranking
    outcsv = "outputs/topK_bestfit_summary.csv"
    pd.DataFrame({
        "var":["USEP"]*K,"day":[best_day]*K,
        "scenario":p_order[:K],"rmse":p_scores[p_order[:K]]
    }).to_csv(outcsv,index=False,mode="w")
    pd.DataFrame({
        "var":["LOAD"]*K,"day":[best_day]*K,
        "scenario":l_order[:K],"rmse":l_scores[l_order[:K]]
    }).to_csv(outcsv,index=False,header=False,mode="a")
    print(f"Wrote ranking to {outcsv}")

    # overlay top-K combined
    for name,data,act,order,ylabel in [
        ("USEP",USEP,act_p,p_order,"USEP (SGD/MWh)"),
        ("LOAD",LOAD,act_l,l_order,"LOAD (MW)")
    ]:
        fig,ax=plt.subplots(figsize=(14,6))
        for s in order[:K]: ax.plot(ts,data[s,day_idx,:],alpha=0.6)
        ax.plot(ts,act,color="k",lw=2.5,label=f"Actual {name}")
        ax.plot(ts,data[order[0],day_idx,:],lw=2.2)
        ax.set_title(f"{name} — Actual vs Top-{K} closest | {best_day}")
        ax.set_xlabel("Time"); ax.set_ylabel(ylabel)
        format_time_axis(ax); ax.legend(); fig.tight_layout()
        fig.savefig(f"outputs/figs/bestfit_{name}_top{K}_{best_day}.png",dpi=150)
        plt.close(fig)

        # now individual top-5 plots
        for j,s in enumerate(order[:args.individual_k]):
            fig,ax=plt.subplots(figsize=(14,6))
            ax.plot(ts,act,color="k",lw=2.5,label="Actual")
            ax.plot(ts,data[s,day_idx,:],lw=2,label=f"Scenario {s}")
            ax.set_title(f"{name} — Scenario {s} vs Actual | {best_day}")
            ax.set_xlabel("Time"); ax.set_ylabel(ylabel)
            format_time_axis(ax); ax.legend(); fig.tight_layout()
            fname=f"outputs/figs/bestfit_{name}_scenario_{j}_{best_day}.png"
            fig.savefig(fname,dpi=150); plt.close(fig)

    print(f"Saved top-{K} overlays and top-{args.individual_k} individual scenario images.")
    print(f"Best-fit day: {best_day}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--scenarios",default="outputs/samples_pt.pkl")
    ap.add_argument("--csv_path",default="~/Real_usep_load_pv_rp.csv")
    ap.add_argument("--date_col",default="DATE")
    ap.add_argument("--usep_col",default="USEP")
    ap.add_argument("--load_col",default="LOAD")
    ap.add_argument("--top_k",type=int,default=15)
    ap.add_argument("--individual_k",type=int,default=5)
    args=ap.parse_args()
    args.csv_path=os.path.expanduser(args.csv_path)
    args.scenarios=os.path.expanduser(args.scenarios)
    main(args)


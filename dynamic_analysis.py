
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats import mannwhitneyu
import math

sns.set(style="whitegrid")
warnings.filterwarnings("ignore", category=UserWarning)

INPUT_DIR = Path(r"C:\Users\sadyk\Downloads\neuromorphic-control-project-main\neuromorphic-control-project-main\results\dynamic_sweep")
OUTPUT_DIR = Path(r"C:\Users\sadyk\Downloads\neuromorphic-control-project-main\neuromorphic-control-project-main\analysis\dynamic_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def safe_load_csv(path: Path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print("Failed to read", path, ":", e)
        return None

def parse_filename_info(name: str):
    base = Path(name).stem
    info = {"motion": None, "speed": None, "radius": None, "seed": None}
    m = re.search(r"results_([^_]+)_sp(\d+)_r(\d+)_seed(\d+)", base)
    if m:
        info["motion"] = m.group(1)
        info["speed"] = float(m.group(2)) / 100.0
        info["radius"] = float(m.group(3)) / 100.0
        info["seed"] = int(m.group(4))
    else:
        if "static" in base: info["motion"] = "static"
        elif "sin" in base: info["motion"] = "sin"
        elif "circle" in base: info["motion"] = "circle"
        elif "random" in base: info["motion"] = "random"
        s = re.search(r"seed(\d+)", base)
        if s: info["seed"] = int(s.group(1))
    return info

def entropy_from_counts(counts):
    counts = np.asarray(counts, dtype=float)
    s = counts.sum()
    if s <= 0: return 0.0
    p = counts / s
    p = p[p>0]
    return -np.sum(p * np.log2(p))

def summarize_episode(df_ep):
    out = {}
    if "endpoint_error" in df_ep.columns:
        out["endpoint_error_mean"] = float(df_ep["endpoint_error"].mean())
        out["endpoint_error_final"] = float(df_ep["endpoint_error"].iloc[-1])
    else:
        out["endpoint_error_mean"] = out["endpoint_error_final"] = np.nan
    out["tracking_error_mean"] = float(df_ep["tracking_error"].mean()) if "tracking_error" in df_ep.columns else np.nan
    out["movement_vigor_mean"] = float(df_ep["movement_vigor"].mean()) if "movement_vigor" in df_ep.columns else np.nan
    out["dopamine_abs_mean"] = float(np.abs(df_ep["dopamine"]).mean()) if "dopamine" in df_ep.columns else np.nan
    if "target" in df_ep.columns:
        cnt = df_ep["target"].value_counts()
        out["selection_entropy"] = float(entropy_from_counts(cnt.values))
        total = cnt.sum()
        out["p_red"] = float(cnt.get("Red",0)/total) if total>0 else np.nan
        out["p_green"] = float(cnt.get("Green",0)/total) if total>0 else np.nan
        out["p_blue"] = float(cnt.get("Blue",0)/total) if total>0 else np.nan
    else:
        out["selection_entropy"] = out["p_red"] = out["p_green"] = out["p_blue"] = np.nan
    # energy (approx)
    if set(("tau1","tau2","dq1","dq2","time")).issubset(df_ep.columns):
        t = df_ep["time"].values
        dt = np.diff(t, prepend=t[0])
        power = np.abs(df_ep["tau1"].values * df_ep["dq1"].values + df_ep["tau2"].values * df_ep["dq2"].values)
        out["energy"] = float(np.nansum(power * dt))
    else:
        out["energy"] = np.nan
    # success final
    if "endpoint_error" in df_ep.columns:
        out["success"] = int(df_ep["endpoint_error"].iloc[-1] < 0.03)
    else:
        out["success"] = np.nan
    # learned values final
    for k in ("q_red","q_green","q_blue"):
        out[k + "_final"] = float(df_ep[k].iloc[-1]) if k in df_ep.columns else np.nan
    return out

def cohens_d(a, b):
    a = np.asarray(a); b = np.asarray(b)
    na, nb = len(a), len(b)
    if na <=1 or nb <=1: return np.nan
    sa = a.var(ddof=1); sb = b.var(ddof=1)
    pooled = math.sqrt(((na-1)*sa + (nb-1)*sb) / (na+nb-2)) if (na+nb-2)>0 else np.nan
    if pooled == 0 or np.isnan(pooled): return np.nan
    return (a.mean() - b.mean()) / pooled

# ---------------------------
# Load data and summarize
# ---------------------------
def load_and_summarize(input_dir: Path):
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError("No CSV files found in " + str(input_dir))
    steps_list = []
    episodes_list = []

    for f in files:
        df = safe_load_csv(f)
        if df is None: continue

        meta = {}
        meta_path = f.with_suffix(f.suffix + ".meta.json")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}

        parsed = parse_filename_info(f.name)
        motion = parsed.get("motion") or meta.get("experiment_config",{}).get("target_motion","unknown")
        speed = parsed.get("speed") or meta.get("experiment_config",{}).get("target_speed", np.nan)
        radius = parsed.get("radius") or meta.get("experiment_config",{}).get("target_radius", np.nan)
        seed = parsed.get("seed") or meta.get("root_seed", None)

        df["_source_file"] = f.name
        df["_motion"] = motion
        df["_speed"] = float(speed) if speed is not None else np.nan
        df["_radius"] = float(radius) if radius is not None else np.nan
        df["_seed"] = seed

        if "episode" not in df.columns:
            df["episode"] = 0

        for ep, df_ep in df.groupby("episode"):
            summ = summarize_episode(df_ep)
            summ.update({"_source_file": f.name, "_motion": motion, "_speed": float(speed), "_radius": float(radius), "_seed": seed, "episode": int(ep)})
            episodes_list.append(summ)

        steps_list.append(df)

    combined_steps = pd.concat(steps_list, ignore_index=True, sort=False) if steps_list else pd.DataFrame()
    df_episodes = pd.DataFrame(episodes_list)
    return combined_steps, df_episodes

# ---------------------------
# Aggregation & comparisons
# ---------------------------
def group_and_compare(df_episodes: pd.DataFrame, output_dir: Path):
    df_episodes.to_csv(output_dir / "episodes_summary.csv", index=False)

    group = df_episodes.groupby(["_motion","_speed","_radius"])
    agg = group.agg({
        "endpoint_error_mean":["mean","std","count"],
        "endpoint_error_final":["mean","std"],
        "tracking_error_mean":["mean","std"],
        "energy":["mean","std"],
        "success":["mean","count"],
        "selection_entropy":["mean","std"]
    })
    agg.columns = ["_".join(col).strip() for col in agg.columns.values]
    agg = agg.reset_index()
    agg.to_csv(output_dir / "group_summary.csv", index=False)

    static_df = df_episodes[df_episodes["_motion"] == "static"]
    dynamic_df = df_episodes[df_episodes["_motion"] != "static"]

    delta_rows = []
    metrics = ["endpoint_error_final","tracking_error_mean","energy","success","selection_entropy"]
    for (motion, speed, radius), grp in dynamic_df.groupby(["_motion","_speed","_radius"]):
        static_grp = static_df[(static_df["_speed"]==speed) & (static_df["_radius"]==radius)]
        if static_grp.empty:
            continue
        for metric in metrics:
            dyn_vals = grp[metric].dropna().values
            stat_vals = static_grp[metric].dropna().values
            if len(dyn_vals)==0 or len(stat_vals)==0:
                continue
            mean_dyn = float(np.mean(dyn_vals))
            mean_stat = float(np.mean(stat_vals))
            delta = mean_dyn - mean_stat
            try:
                U, p = mannwhitneyu(dyn_vals, stat_vals, alternative='two-sided')
            except Exception:
                U, p = np.nan, np.nan
            cd = cohens_d(dyn_vals, stat_vals)
            delta_rows.append({
                "motion": motion,
                "speed": speed,
                "radius": radius,
                "metric": metric,
                "mean_dynamic": mean_dyn,
                "mean_static": mean_stat,
                "delta": delta,
                "mannwhitney_u": U,
                "p_raw": p,
                "cohen_d": cd,
                "n_dynamic": len(dyn_vals),
                "n_static": len(stat_vals)
            })
    deltas = pd.DataFrame(delta_rows)
    if not deltas.empty:
        m = len(deltas)
        deltas["p_adj"] = deltas["p_raw"].apply(lambda x: min(1.0, x*m) if not (np.isnan(x) or x is None) else np.nan)
    else:
        deltas["p_adj"] = []

    deltas.to_csv(output_dir / "deltas.csv", index=False)

    return agg, deltas

# ---------------------------
# Consolidated plotting utilities
# ---------------------------
def plot_combined_boxplots(df_episodes, output_dir: Path):
    metrics = [
        ("endpoint_error_final", "Endpoint error (final)"),
        ("tracking_error_mean", "Tracking error (mean)"),
        ("selection_entropy", "Selection entropy"),
        ("dopamine_abs_mean", "Dopamine |δ| (mean)"),
        ("movement_vigor_mean", "Movement vigor (mean)"),
        ("energy", "Energy (approx)")
    ]
    n = len(metrics)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
    axes = axes.flatten()
    for i, (metric, title) in enumerate(metrics):
        ax = axes[i]
        if metric not in df_episodes.columns:
            ax.axis("off")
            continue
        sns.boxplot(x="_motion", y=metric, hue="_speed", data=df_episodes, showfliers=False, ax=ax)
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=25)
    for j in range(i+1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "boxplots_main_metrics_by_motion_speed.png", dpi=200)
    plt.close()

def plot_endpoint_vs_speed(df_episodes, output_dir: Path):
    plt.figure(figsize=(10,6))
    motions = sorted(df_episodes["_motion"].unique())
    for motion in motions:
        dfm = df_episodes[df_episodes["_motion"]==motion]
        if dfm.empty: continue
        by_speed = dfm.groupby("_speed")["endpoint_error_final"].agg(["mean","std"]).reset_index().sort_values("_speed")
        plt.errorbar(by_speed["_speed"], by_speed["mean"], yerr=by_speed["std"], label=motion, marker='o')
    plt.xlabel("Speed (cycles/sec)")
    plt.ylabel("Endpoint error (final)")
    plt.title("Endpoint error vs speed (per motion)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "endpoint_vs_speed.png", dpi=200)
    plt.close()

def plot_success_rate_bars(df_episodes, output_dir: Path):
    agg = df_episodes.groupby("_motion")["success"].mean().reset_index()
    plt.figure(figsize=(8,4))
    sns.barplot(x="_motion", y="success", data=agg)
    plt.ylabel("Success rate (mean across episodes)")
    plt.title("Success rate by motion (aggregate)")
    plt.tight_layout()
    plt.savefig(output_dir / "bar_success_by_motion.png", dpi=200)
    plt.close()

def plot_learned_values(df_episodes, output_dir: Path):
    learned_keys = [c for c in ("q_red_final","q_green_final","q_blue_final") if c in df_episodes.columns]
    if not learned_keys:
        return
    plt.figure(figsize=(9,5))
    melted = df_episodes.melt(id_vars=["_motion","_speed"], value_vars=learned_keys, var_name="value", value_name="learned")
    sns.boxplot(x="_motion", y="learned", hue="value", data=melted, showfliers=False)
    plt.title("Learned values (final) by motion")
    plt.tight_layout()
    plt.savefig(output_dir / "learned_values_box.png", dpi=200)
    plt.close()

def plot_delta_heatmaps_grid(deltas: pd.DataFrame, output_dir: Path, metric="endpoint_error_final"):
    """
    Consolidated heatmap figure: each subplot is a motion with speed x radius grid showing delta.
    """
    if deltas.empty:
        return
    motions = sorted(deltas["motion"].unique())
    if not motions:
        return

    # Determine grid layout
    cols = min(3, len(motions))
    rows = math.ceil(len(motions) / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.0))
    axs = axs.flatten()
    for i, motion in enumerate(motions):
        ax = axs[i]
        dfm = deltas[(deltas["motion"]==motion) & (deltas["metric"]==metric)]
        if dfm.empty:
            ax.axis("off")
            continue
        # pivot table with speeds as rows and radii as columns
        pivot = dfm.pivot(index="speed", columns="radius", values="delta")
        # sort axes
        pivot = pivot.sort_index().reindex(sorted(pivot.columns), axis=1)
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdBu_r", center=0, ax=ax, cbar=(i==len(motions)-1))
        ax.set_title(motion)
        ax.set_xlabel("radius (m)")
        ax.set_ylabel("speed (cycles/s)")
    for j in range(i+1, len(axs)):
        axs[j].axis("off")
    plt.suptitle(f"Delta (dynamic - static) of {metric} — per-motion grid", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f"heatmap_delta_{metric}_grid.png", dpi=200, bbox_inches="tight")
    plt.close()

def plot_cohend_heatmaps_grid(deltas: pd.DataFrame, output_dir: Path, metric="endpoint_error_final"):
    if deltas.empty:
        return
    motions = sorted(deltas["motion"].unique())
    if not motions:
        return

    cols = min(3, len(motions))
    rows = math.ceil(len(motions) / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.0))
    axs = axs.flatten()
    for i, motion in enumerate(motions):
        ax = axs[i]
        dfm = deltas[(deltas["motion"]==motion) & (deltas["metric"]==metric)]
        if dfm.empty:
            ax.axis("off")
            continue
        pivot = dfm.pivot(index="speed", columns="radius", values="cohen_d")
        pivot = pivot.sort_index().reindex(sorted(pivot.columns), axis=1)
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, cbar=(i==len(motions)-1))
        ax.set_title(motion)
        ax.set_xlabel("radius (m)")
        ax.set_ylabel("speed (cycles/s)")
    for j in range(i+1, len(axs)):
        axs[j].axis("off")
    plt.suptitle(f"Cohen's d for {metric} (dynamic vs static) — per-motion grid", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f"heatmap_cohend_{metric}_grid.png", dpi=200, bbox_inches="tight")
    plt.close()

# ---------------------------
# Main pipeline
# ---------------------------
def main():
    print("Loading from:", INPUT_DIR.resolve())
    combined_steps, df_episodes = load_and_summarize(INPUT_DIR)
    if combined_steps is None or df_episodes is None or df_episodes.empty:
        print("No data found or summarization failed.")
        return

    combined_steps.to_csv(OUTPUT_DIR / "combined_steps.csv", index=False)
    print("Saved combined_steps.csv, episodes_summary.csv next...")

    agg, deltas = group_and_compare(df_episodes, OUTPUT_DIR)

    # Consolidated plots
    plot_combined_boxplots(df_episodes, OUTPUT_DIR)
    plot_endpoint_vs_speed(df_episodes, OUTPUT_DIR)
    plot_success_rate_bars(df_episodes, OUTPUT_DIR)
    plot_learned_values(df_episodes, OUTPUT_DIR)

    # Consolidated heatmap grids (endpoint deltas & Cohen's d)
    if not deltas.empty:
        plot_delta_heatmaps_grid(deltas, OUTPUT_DIR, metric="endpoint_error_final")
        plot_cohend_heatmaps_grid(deltas, OUTPUT_DIR, metric="endpoint_error_final")

    # Save deltas and group summaries already handled
    print("Saved summaries and consolidated plots to:", OUTPUT_DIR.resolve())
    print("Key files:")
    for name in ["combined_steps.csv","episodes_summary.csv","group_summary.csv","deltas.csv","boxplots_main_metrics_by_motion_speed.png","heatmap_delta_endpoint_error_final_grid.png","heatmap_cohend_endpoint_error_final_grid.png"]:
        print(" -", OUTPUT_DIR / name)

if __name__ == "__main__":
    main()
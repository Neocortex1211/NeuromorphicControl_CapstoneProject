
import argparse
from pathlib import Path
import json
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats import kruskal
from itertools import combinations

sns.set(style="whitegrid")
warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_ENDPOINT_THRESHOLD = 0.03  # success threshold (meters)

FORCED_OUTPUT_DIR = Path(r"C:\Users\sadyk\Downloads\neuromorphic-control-project-main\neuromorphic-control-project-main\analysis\lesion_analysis")


def entropy_from_counts(counts):
    counts = np.asarray(counts, dtype=float)
    s = counts.sum()
    if s <= 0:
        return 0.0
    probs = counts / s
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def extract_meta_from_sidecar(data_path: Path):
    candidates = [
        data_path.with_suffix(data_path.suffix + ".meta.json"),
        data_path.with_suffix(".meta.json"),
        data_path.with_suffix(".json"),
    ]
    for c in candidates:
        if c.exists():
            try:
                return json.loads(c.read_text(encoding="utf-8"))
            except Exception:
                return None
    return None


def parse_run_info_from_filename(fname: str):
    base = Path(fname).stem.lower()
    info = {"lesion": None, "severity": None, "seed": None}
    m = re.search(r"seed(\d+)", base)
    if m:
        info["seed"] = int(m.group(1))
    for sev in ("mild", "moderate", "severe"):
        if sev in base:
            info["severity"] = sev
            break
    if "pfc" in base or "prefrontal" in base:
        info["lesion"] = "pfc"
    elif "parietal" in base:
        info["lesion"] = "parietal"
    elif "basal" in base or "bg" in base:
        info["lesion"] = "basal_ganglia"
    elif "cerebell" in base or "cb" in base:
        info["lesion"] = "cerebellum"
    elif "vision" in base:
        info["lesion"] = "vision"
    else:
        if "none" in base or "control" in base:
            info["lesion"] = "none"
    return info


def safe_load_table(path: Path):
    try:
        if path.suffix.lower() in (".csv",):
            return pd.read_csv(path)
        elif path.suffix.lower() in (".xls", ".xlsx"):
            return pd.read_excel(path)
        else:
            try:
                return pd.read_csv(path)
            except Exception:
                return pd.read_excel(path)
    except Exception as e:
        print("Failed to load", path, ":", e)
        return None


def summarize_episode(df_episode, endpoint_thresh=DEFAULT_ENDPOINT_THRESHOLD):
    out = {}
    if "endpoint_error" in df_episode.columns:
        out["endpoint_error_mean"] = float(df_episode["endpoint_error"].mean())
        out["endpoint_error_final"] = float(df_episode["endpoint_error"].iloc[-1])
    else:
        out["endpoint_error_mean"] = out["endpoint_error_final"] = np.nan
    out["tracking_error_mean"] = float(df_episode["tracking_error"].mean()) if "tracking_error" in df_episode.columns else np.nan
    out["dopamine_mean"] = float(df_episode["dopamine"].mean()) if "dopamine" in df_episode.columns else np.nan
    out["dopamine_abs_mean"] = float(np.abs(df_episode["dopamine"]).mean()) if "dopamine" in df_episode.columns else np.nan
    out["movement_vigor_mean"] = float(df_episode["movement_vigor"].mean()) if "movement_vigor" in df_episode.columns else np.nan
    if "target" in df_episode.columns:
        counts = df_episode["target"].value_counts()
        out["selection_entropy"] = float(entropy_from_counts(counts.values))
        total = counts.sum()
        out["p_red"] = float(counts.get("Red", 0) / total) if total > 0 else np.nan
        out["p_green"] = float(counts.get("Green", 0) / total) if total > 0 else np.nan
        out["p_blue"] = float(counts.get("Blue", 0) / total) if total > 0 else np.nan
    else:
        out["selection_entropy"] = out["p_red"] = out["p_green"] = out["p_blue"] = np.nan
    for k in ("q_red", "q_green", "q_blue"):
        if k in df_episode.columns:
            out[f"{k}_final"] = float(df_episode[k].iloc[-1])
        else:
            out[f"{k}_final"] = np.nan
    if set(("tau1", "tau2", "dq1", "dq2", "time")).issubset(df_episode.columns):
        t = df_episode["time"].values
        dt = np.diff(t, prepend=t[0])
        power = np.abs(df_episode["tau1"].values * df_episode["dq1"].values + df_episode["tau2"].values * df_episode["dq2"].values)
        out["energy"] = float(np.nansum(power * dt))
    else:
        out["energy"] = np.nan
    if "endpoint_error" in df_episode.columns:
        out["success"] = int(df_episode["endpoint_error"].iloc[-1] < endpoint_thresh)
    else:
        out["success"] = np.nan
    if "movement_start_time" in df_episode.columns and "time" in df_episode.columns:
        try:
            start = float(df_episode["movement_start_time"].iloc[0])
            end = float(df_episode["time"].iloc[-1])
            out["movement_time"] = float(max(0.0, end - start))
        except Exception:
            out["movement_time"] = np.nan
    else:
        out["movement_time"] = np.nan
    return out


def find_input_files(input_dir: Path):
    patterns = ("*.csv", "*.xls", "*.xlsx")
    files = []
    for pat in patterns:
        files.extend(list(input_dir.rglob(pat)))
    files = sorted(set(files))
    return files


def _save_csv(df, outpath: Path, name: str):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outpath, index=False)
    print("Saved:", outpath)


def analyze_folder(input_dir: Path, output_dir: Path = None, timeseries_severity: str = None):
    # If input_dir not provided, try sensible fallbacks
    tried = []
    if input_dir is None:
        input_dir = Path.cwd() / "results"
    input_dir = Path(input_dir)

    fallback_candidates = [
        input_dir,
        Path.cwd() / "results",
        Path.cwd() / "results" / "lesion_sweep",
        Path(__file__).resolve().parent / "results",
        Path(__file__).resolve().parent / "results" / "lesion_sweep",
    ]

    found_files = []
    used_input = None
    for candidate in fallback_candidates:
        tried.append(str(candidate))
        if candidate.exists() and candidate.is_dir():
            files = find_input_files(candidate)
            if files:
                found_files = files
                used_input = candidate
                break

    if not found_files:
        if input_dir.exists():
            found_files = find_input_files(input_dir)

    if not found_files:
        print("No CSV/XLSX files found. Attempted the following directories (in order):")
        for p in tried:
            print(" -", p)
        print("You can pass --input <path> to point the script at your results folder.")
        raise FileNotFoundError("No CSV/XLSX files found in tried locations.")


    if output_dir is None:
        output_dir = FORCED_OUTPUT_DIR
    else:
        output_dir = FORCED_OUTPUT_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_steps = []
    episodes_rows = []

    for csv_path in found_files:
        print("Loading:", csv_path)
        df = safe_load_table(csv_path)
        if df is None:
            continue

        meta = extract_meta_from_sidecar(csv_path) or {}
        lesion_label = None
        severity_label = None
        seed_val = None
        if "lesion_profile_map" in meta:
            lmap = meta.get("lesion_profile_map", {})
            for k, v in lmap.items():
                if v is not None:
                    lesion_label = "pfc" if k == "prefrontal" else k
                    if isinstance(v, dict):
                        gs = v.get("gain_scale", None)
                        ns = v.get("noise_std", None)
                        if gs is not None:
                            if gs >= 0.75:
                                severity_label = "mild"
                            elif gs >= 0.5:
                                severity_label = "moderate"
                            else:
                                severity_label = "severe"
                        elif ns is not None:
                            severity_label = "moderate" if ns < 0.5 else "severe"
                    break
        parsed = parse_run_info_from_filename(csv_path.name)
        if lesion_label is None:
            lesion_label = parsed.get("lesion") or meta.get("experiment_config", {}).get("lesion", "none")
        if severity_label is None:
            severity_label = parsed.get("severity") or "default"
        if seed_val is None:
            seed_val = parsed.get("seed") or meta.get("root_seed")

        df["_source_file"] = csv_path.name
        df["_lesion"] = lesion_label
        df["_severity"] = severity_label
        df["_seed"] = seed_val

        if "episode" not in df.columns:
            df["episode"] = 0

        for (ep_key), df_ep in df.groupby("episode"):
            summary = summarize_episode(df_ep)
            summary.update({
                "_source_file": csv_path.name,
                "_lesion": lesion_label,
                "_severity": severity_label,
                "_seed": seed_val,
                "episode": int(ep_key)
            })
            episodes_rows.append(summary)

        all_steps.append(df)

    combined_steps = pd.concat(all_steps, ignore_index=True, sort=False)
    _save_csv(combined_steps, output_dir / "combined_steps.csv", "combined_steps")

    df_episodes = pd.DataFrame(episodes_rows)
    _save_csv(df_episodes, output_dir / "episodes_summary.csv", "episodes_summary")


    group = df_episodes.groupby(["_lesion", "_severity"])
    agg = group.agg({
        "endpoint_error_mean": ["mean", "std", "count"],
        "endpoint_error_final": ["mean", "std"],
        "tracking_error_mean": ["mean", "std"],
        "movement_vigor_mean": ["mean", "std"],
        "dopamine_abs_mean": ["mean", "std"],
        "selection_entropy": ["mean", "std"],
        "energy": ["mean", "std"],
        "success": ["mean", "count"]
    })
    agg.columns = ["_".join(col).strip() for col in agg.columns.values]
    agg = agg.reset_index()
    _save_csv(agg, output_dir / "group_summary.csv", "group_summary")


    if "target" in combined_steps.columns:
        sel_counts = combined_steps.groupby(["_lesion", "_severity", "target"]).size().reset_index(name="count")
        _save_csv(sel_counts, output_dir / "selection_counts_by_lesion_severity.csv", "selection_counts_by_lesion_severity")

        # aggregated across severity -> proportions per lesion
        agg_sel = sel_counts.groupby(["_lesion", "target"])["count"].sum().reset_index()
        pivot = agg_sel.pivot(index="_lesion", columns="target", values="count").fillna(0).astype(int)
        pivot["total"] = pivot.sum(axis=1)
        prop = pivot.div(pivot["total"], axis=0).drop(columns=["total"])
        _save_csv(pivot.reset_index(), output_dir / "selection_counts_by_lesion.csv", "selection_counts_by_lesion")
        _save_csv(prop.reset_index(), output_dir / "selection_proportions_by_lesion.csv", "selection_proportions_by_lesion")

        # Stacked bar (proportions) aggregated across severity
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            prop_plot = prop.copy()
            cols = [c for c in ["Red", "Green", "Blue"] if c in prop_plot.columns]
            prop_plot[cols].plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
            ax.set_ylabel("Proportion of selections")
            ax.set_title("Selection proportions by lesion (aggregated across severity)")
            ax.legend(title="Target", bbox_to_anchor=(1.02, 1), loc="upper left")
            plt.tight_layout()
            plt.savefig(output_dir / "selection_proportions_by_lesion_stacked.png", dpi=200)
            plt.close()
        except Exception as e:
            print("Failed to create stacked selection proportions plot:", e)

        # Grouped bar (counts) per lesion x severity (single figure)
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            total_by_lesion_sev = sel_counts.groupby(["_lesion", "_severity"])["count"].sum().reset_index()
            sns.barplot(x="_lesion", y="count", hue="_severity", data=total_by_lesion_sev, ax=ax)
            ax.set_title("Selection counts by lesion and severity (total selections per run set)")
            ax.set_ylabel("Selection counts (sum across targets)")
            plt.tight_layout()
            plt.savefig(output_dir / "selection_counts_by_lesion_severity_bar.png", dpi=200)
            plt.close()
        except Exception as e:
            print("Failed to create grouped selection counts plot:", e)
    else:
        print("No 'target' column present in combined_steps; skipping selection counts/proportion outputs.")

    # ------------------------
    # Inferential tests (Kruskal across lesions)
    # ------------------------
    metrics = [
        "endpoint_error_final",
        "tracking_error_mean",
        "selection_entropy",
        "dopamine_abs_mean",
        "movement_vigor_mean",
        "energy",
    ]

    infer_rows = []
    lesion_groups = sorted(df_episodes["_lesion"].unique()) if "_lesion" in df_episodes.columns else []
    for metric in metrics:
        groups = []
        for lg in lesion_groups:
            vals = df_episodes.loc[df_episodes["_lesion"] == lg, metric].dropna().values
            groups.append(vals)
        nonempty = [g for g in groups if len(g) > 0]
        if len(nonempty) < 2:
            continue
        try:
            H, p = kruskal(*[g for g in groups if len(g) > 0])
            infer_rows.append({"metric": metric, "test": "kruskal", "H": float(H), "p": float(p)})
        except Exception as e:
            infer_rows.append({"metric": metric, "test": "kruskal", "error": str(e)})

    pd.DataFrame(infer_rows).to_csv(output_dir / "inferential_results.csv", index=False)

    # ------------------------
    # Consolidated boxplots (by lesion, hue severity) - keep a few clear metrics
    # ------------------------
    plot_metrics = [
        ("endpoint_error_final", "Endpoint error (final)"),
        ("tracking_error_mean", "Tracking error (mean)"),
        ("selection_entropy", "Selection entropy"),
        ("dopamine_abs_mean", "Dopamine |δ| (mean)"),
        ("movement_vigor_mean", "Movement vigor (mean)"),
        ("energy", "Energy (approx)")
    ]

    # One consolidated figure for boxplots (grid)
    try:
        n = len(plot_metrics)
        cols = 2
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
        axes = axes.flatten()
        for i, (metric, title) in enumerate(plot_metrics):
            ax = axes[i]
            if metric not in df_episodes.columns:
                ax.axis("off")
                continue
            sns.boxplot(x="_lesion", y=metric, hue="_severity", data=df_episodes, showfliers=False, ax=ax)
            ax.set_title(title)
            ax.tick_params(axis='x', rotation=25)
        # hide extra axes
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        plt.savefig(output_dir / "boxplots_main_metrics_by_lesion_severity.png", dpi=200)
        plt.close()
    except Exception as e:
        print("Failed to create consolidated boxplots:", e)

    # Learned-values boxplot (final) (single figure)
    learned_keys = [k for k in ("q_red_final", "q_green_final", "q_blue_final") if k in df_episodes.columns]
    if learned_keys:
        try:
            plt.figure(figsize=(9, 5))
            melted = df_episodes.melt(id_vars=["_lesion", "_severity"], value_vars=learned_keys, var_name="value", value_name="learned")
            sns.boxplot(x="_lesion", y="learned", hue="value", data=melted, showfliers=False)
            plt.title("Learned values (final) by lesion")
            plt.xticks(rotation=20)
            plt.tight_layout()
            plt.savefig(output_dir / "learned_values_box_by_lesion.png", dpi=200)
            plt.close()
        except Exception as e:
            print("Failed to create learned-values boxplot:", e)

    # ------------------------
    # Time series plots (consolidated)
    # ------------------------
    print("Time-series severity focus (ignored for consolidated plots):", timeseries_severity)

    # Ensure we have a step index.
    if "step" in combined_steps.columns:
        combined_steps["step_idx"] = combined_steps["step"].astype(int)
    else:
        combined_steps = combined_steps.sort_values(["_source_file", "_seed", "episode"]).reset_index(drop=True)
        combined_steps["_row_in_episode"] = combined_steps.groupby(["_source_file", "_seed", "episode"]).cumcount()
        combined_steps["step_idx"] = combined_steps["_row_in_episode"]

    ts_dir = output_dir / "time_series"
    ts_dir.mkdir(parents=True, exist_ok=True)

    def plot_consolidated_ts(metric, ylabel, fname, smooth=5):
        if metric not in combined_steps.columns and metric not in ("dopamine_abs",):
            print("Skipping time-series: metric not present:", metric)
            return
        df_ts = combined_steps.copy()
        if metric == "dopamine_abs":
            df_ts["dopamine_abs"] = df_ts["dopamine"].abs() if "dopamine" in df_ts.columns else np.nan
        # group by lesion and step_idx, compute mean/std
        grouped = df_ts.groupby(["_lesion", "step_idx"])[metric].agg(["mean", "std"]).reset_index()
        lesions = sorted(grouped["_lesion"].unique())
        if not lesions:
            print("No lesions found for time-series metric:", metric)
            return
        plt.figure(figsize=(10, 5))
        for lesion in lesions:
            g = grouped[grouped["_lesion"] == lesion]
            if g.empty:
                continue
            x = g["step"].values if "step" in g.columns else g["step_idx"].values
            mean = pd.Series(g["mean"].values).rolling(smooth, min_periods=1, center=True).mean().values
            std = pd.Series(g["std"].fillna(0).values).rolling(smooth, min_periods=1, center=True).mean().values
            plt.plot(x, mean, label=str(lesion))
            plt.fill_between(x, mean - std, mean + std, alpha=0.15)
        plt.xlabel("Step")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} — mean ± std by lesion (consolidated)")
        plt.legend(loc="upper right", fontsize="small")
        plt.tight_layout()
        plt.savefig(ts_dir / fname, dpi=200)
        plt.close()

    # (A) Endpoint error time-series (consolidated)
    plot_consolidated_ts("endpoint_error", "Endpoint error (per-step)", "ts_endpoint_error_by_lesion_consolidated.png")
    # (B) Dopamine abs
    plot_consolidated_ts("dopamine_abs", "Dopamine |δ| (per-step)", "ts_dopamine_abs_by_lesion_consolidated.png")
    # (C) Decision confidence
    plot_consolidated_ts("decision_confidence", "Decision confidence (per-step)", "ts_decision_confidence_by_lesion_consolidated.png")

    print("Analysis complete. Outputs (forced to):")
    print(" -", output_dir / "combined_steps.csv")
    print(" -", output_dir / "episodes_summary.csv")
    print(" -", output_dir / "group_summary.csv")
    print(" -", output_dir / "inferential_results.csv")
    print(" - selection counts/proportions and consolidated plots in", output_dir)
    print(" - time series directory:", ts_dir)

    return {
        "combined_steps": output_dir / "combined_steps.csv",
        "episodes_summary": output_dir / "episodes_summary.csv",
        "group_summary": output_dir / "group_summary.csv",
        "inferential": output_dir / "inferential_results.csv",
        "selection_counts": output_dir / "selection_counts_by_lesion.csv",
        "selection_proportions": output_dir / "selection_proportions_by_lesion.csv",
        "time_series_dir": ts_dir
    }


def main():
    p = argparse.ArgumentParser(description="Analyze lesion experiment CSV/XLSX outputs and produce summary, consolidated plots and grouped time-series.")
    p.add_argument("--input", "-i", type=str, default=None, help="Input folder containing CSV/XLSX result files (and .meta.json sidecars). If omitted several fallback folders are tried.")
    p.add_argument("--output", "-o", type=str, default=str(FORCED_OUTPUT_DIR), help="(Ignored) Output folder for summary CSVs and plots - outputs are forced to the project analysis folder.")
    p.add_argument("--severity", "-s", type=str, default=None, help="Preferred severity label for time-series (e.g. severe, moderate, mild). If omitted, the script picks 'severe' if present.")
    args = p.parse_args()

    input_dir = Path(args.input) if args.input else None
    output_dir = Path(args.output)
    try:
        analyze_folder(input_dir, output_dir, timeseries_severity=args.severity)
    except FileNotFoundError as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
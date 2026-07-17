
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, levene, f_oneway, kruskal, mannwhitneyu
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations
import math
import warnings
import sys

sns.set(style="whitegrid")
warnings.filterwarnings("ignore", category=UserWarning)

INPUT_DIR = Path(r"C:/Users/sadyk/Downloads/neuromorphic-control-project-main/neuromorphic-control-project-main/results")
OUTPUT_DIR = Path(r"C:/Users/sadyk/Downloads/neuromorphic-control-project-main/neuromorphic-control-project-main/analysis")
ENDPOINT_THRESHOLD = 0.03  # success threshold in meters

# ---------- Helpers ----------
def eta_squared(groups):
    all_values = np.concatenate(groups)
    grand_mean = np.mean(all_values)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = np.sum((all_values - grand_mean) ** 2)
    return ss_between / ss_total if ss_total > 0 else 0.0

def cohens_d(a, b):
    a = np.asarray(a); b = np.asarray(b)
    na, nb = len(a), len(b)
    if na + nb - 2 <= 0:
        return np.nan
    pooled = math.sqrt(((na-1)*a.var(ddof=1) + (nb-1)*b.var(ddof=1)) / (na+nb-2))
    if pooled == 0:
        return np.nan
    return (a.mean() - b.mean()) / pooled

def entropy_from_counts(counts):
    counts = np.asarray(counts)
    probs = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0.0

# ---------- Load files & tag condition ----------
def load_condition_files(input_dir: Path):
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    frames = []
    for f in files:
        name = f.name.lower()
        cond = None
        if "reward" in name:
            cond = "Reward"
        elif "energy" in name:
            cond = "Energy"
        elif "explore" in name or "exploration" in name:
            cond = "Exploration"
        else:
            # skip unrelated CSVs
            continue
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print("Warning: failed to read", f, ":", e)
            continue
        # attach metadata
        df["_condition"] = cond
        df["_source_file"] = f.name
        # try to parse seed from filename
        import re
        m = re.search(r"seed(\d+)", f.name)
        df["_seed"] = int(m.group(1)) if m else None
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No condition CSVs (reward/energy/explore) found in " + str(input_dir))
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined

# ---------- Episode summarization ----------
def summarize_episode(df_episode, endpoint_thresh=ENDPOINT_THRESHOLD):
    out = {}
    # endpoint errors
    if "endpoint_error" in df_episode.columns:
        out["endpoint_error_mean"] = float(df_episode["endpoint_error"].mean())
        out["endpoint_error_final"] = float(df_episode["endpoint_error"].iloc[-1])
    else:
        out["endpoint_error_mean"] = out["endpoint_error_final"] = np.nan
    out["tracking_error_mean"] = float(df_episode["tracking_error"].mean()) if "tracking_error" in df_episode.columns else np.nan
    # dopamine
    out["dopamine_mean"] = float(df_episode["dopamine"].mean()) if "dopamine" in df_episode.columns else np.nan
    out["dopamine_abs_mean"] = float(np.abs(df_episode["dopamine"]).mean()) if "dopamine" in df_episode.columns else np.nan
    # vigor
    out["movement_vigor_mean"] = float(df_episode["movement_vigor"].mean()) if "movement_vigor" in df_episode.columns else np.nan
    # selection entropy and probs
    if "target" in df_episode.columns:
        counts = df_episode["target"].value_counts()
        out["selection_entropy"] = float(entropy_from_counts(counts.values))
        total = counts.sum()
        out["p_red"] = float(counts.get("Red", 0) / total) if total>0 else np.nan
        out["p_green"] = float(counts.get("Green", 0) / total) if total>0 else np.nan
        out["p_blue"] = float(counts.get("Blue", 0) / total) if total>0 else np.nan
    else:
        out["selection_entropy"] = out["p_red"] = out["p_green"] = out["p_blue"] = np.nan
    # learned values (final)
    for k in ("q_red","q_green","q_blue"):
        if k in df_episode.columns:
            out[f"{k}_final"] = float(df_episode[k].iloc[-1])
        else:
            out[f"{k}_final"] = np.nan
    # energy
    if set(("tau1","tau2","dq1","dq2","time")).issubset(df_episode.columns):
        t = df_episode["time"].values
        dt = np.diff(t, prepend=t[0])
        power = np.abs(df_episode["tau1"].values * df_episode["dq1"].values + df_episode["tau2"].values * df_episode["dq2"].values)
        out["energy"] = float(np.sum(power * dt))
    else:
        out["energy"] = np.nan
    # success
    if "endpoint_error" in df_episode.columns:
        out["success"] = int(df_episode["endpoint_error"].iloc[-1] < endpoint_thresh)
    else:
        out["success"] = np.nan
    # movement time (best-effort)
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

def aggregate_per_episode(df_all):
    keys = ["_condition"]
    if "_seed" in df_all.columns:
        keys.append("_seed")
    if "episode" in df_all.columns:
        keys.append("episode")
    else:
        keys.append("_source_file")
    rows = []
    for group_vals, df_grp in df_all.groupby(keys):
        meta = {}
        if isinstance(group_vals, tuple):
            for k,v in zip(keys, group_vals):
                meta[k] = v
        else:
            meta[keys[0]] = group_vals
        summary = summarize_episode(df_grp)
        rows.append({**meta, **summary})
    return pd.DataFrame(rows)

# ---------- Robust testing for each metric ----------
def robust_test(metric, df_episodes, conds):
    results = {"metric": metric}
    # collect groups
    groups_vals = []
    groups_n = []
    for c in conds:
        arr = df_episodes.loc[df_episodes["_condition"]==c, metric].dropna().values
        groups_vals.append(arr)
        groups_n.append(len(arr))
    results["counts"] = dict(zip(conds, groups_n))
    # if fewer than two groups with data -> skip
    if sum(n>0 for n in groups_n) < 2:
        results["note"] = "insufficient data"
        return results, []
    # check if all numbers identical (degenerate) -> skip
    all_vals = np.concatenate([g for g in groups_vals if len(g)>0])
    if np.nanmax(all_vals) == np.nanmin(all_vals):
        results["note"] = "all values identical across groups"
        return results, []
    normal = True
    sh_p = {}
    try:
        for i,c in enumerate(conds):
            arr = groups_vals[i]
            if len(arr) >= 3 and len(arr) <= 5000:
                p = shapiro(arr).pvalue
                sh_p[c] = float(p)
                if p < 0.05:
                    normal = False
            else:
                sh_p[c] = None
    except Exception:
        normal = False
    
    try:
        lev_p = float(levene(*[g for g in groups_vals if len(g)>0]).pvalue)
    except Exception:
        lev_p = None
    results.update({"shapiro_p": sh_p, "levene_p": lev_p})
    posthoc = []
    # Decide test
    try:
        if normal and (lev_p is None or lev_p >= 0.05):
            # ANOVA
            valid_groups = [g for g in groups_vals if len(g)>0]
            F, p = f_oneway(*valid_groups)
            results.update({"test":"ANOVA","F":float(F),"p":float(p)})
            # Tukey HSD - need concatenated array and labels
            concat = []
            labels = []
            for c, g in zip(conds, groups_vals):
                concat.extend(g.tolist())
                labels.extend([c]*len(g))
            if len(concat) > 0:
                try:
                    tuk = pairwise_tukeyhsd(endog=np.array(concat), groups=np.array(labels), alpha=0.05)
                    for row in tuk.summary().data[1:]:
                        posthoc.append({"metric":metric,"group1":row[0],"group2":row[1],"mean_diff":float(row[2]),"p_adj":float(row[3]),"reject":bool(row[6])})
                except Exception:
                    pass
        else:
            # Kruskal-Wallis non-parametric
            valid_groups = [g for g in groups_vals if len(g)>0]
            if len(valid_groups) >= 2:
                try:
                    H, p = kruskal(*valid_groups)
                    results.update({"test":"Kruskal","H":float(H),"p":float(p)})
                    # pairwise Mann-Whitney with Bonferroni correction
                    for (i,a),(j,b) in combinations(list(enumerate(conds)),2):
                        g1 = groups_vals[i]; g2 = groups_vals[j]
                        if len(g1)==0 or len(g2)==0: continue
                        u, pu = mannwhitneyu(g1,g2,alternative='two-sided')
                        bonf = min(1.0, pu * (len(conds)*(len(conds)-1)/2))
                        posthoc.append({"metric":metric,"group1":a,"group2":b,"u":float(u),"p_raw":float(pu),"p_adj":float(bonf)})
                except ValueError:
                    results["note"] = "kruskal failed: possible identical values"
            else:
                results["note"] = "not enough non-empty groups for Kruskal"
    except Exception as e:
        results["error"] = str(e)
    
    for (i,a),(j,b) in combinations(list(enumerate(conds)),2):
        g1 = groups_vals[i]; g2 = groups_vals[j]
        if len(g1)==0 or len(g2)==0: continue
        d = cohens_d(g1, g2)
        posthoc.append({"metric":metric,"group1":a,"group2":b,"cohen_d":float(d) if not np.isnan(d) else None})
    return results, posthoc

# ---------- Main ----------
def main():
    print("Input directory:", INPUT_DIR.resolve())
    if not INPUT_DIR.exists():
        print("ERROR: input folder not found:", INPUT_DIR.resolve())
        return
    try:
        df_steps = load_condition_files(INPUT_DIR)
    except FileNotFoundError as e:
        print("Error loading data:", e)
        return
    print("Loaded step-level rows:", len(df_steps))
    df_episodes = aggregate_per_episode(df_steps)
    print("Aggregated episode-level rows:", len(df_episodes))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_episodes.to_csv(OUTPUT_DIR / "combined_per_condition.csv", index=False)
    # metrics to analyze (only those present)
    candidate_metrics = [
        "endpoint_error_final","endpoint_error_mean","tracking_error_mean",
        "dopamine_abs_mean","movement_vigor_mean","energy","movement_time","selection_entropy"
    ]
    metrics = [m for m in candidate_metrics if m in df_episodes.columns]
    # ensure we have condition labels
    conds = sorted(df_episodes["_condition"].unique())
    print("Detected conditions:", conds)
    all_results = []
    all_posthoc = []
    for metric in metrics:
        res, post = robust_test(metric, df_episodes, conds)
        all_results.append(res)
        all_posthoc.extend(post)
        print(f"Metric: {metric} -> {res.get('test', res.get('note','no test'))}")
    # Save tables
    pd.DataFrame(all_results).to_csv(OUTPUT_DIR / "inferential_summary.csv", index=False)
    if len(all_posthoc):
        pd.DataFrame(all_posthoc).to_csv(OUTPUT_DIR / "posthoc_results.csv", index=False)
    # Produce boxplots for metrics (safe)
    for metric in metrics:
        plt.figure(figsize=(6,4))
        try:
            sns.boxplot(x="_condition", y=metric, data=df_episodes, order=conds, showfliers=False)
            plt.title(metric.replace("_"," ").title())
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"box_{metric}.png", dpi=200)
            plt.close()
        except Exception as e:
            print("Plot failed for", metric, e)
    # Plot learned values final by condition if present
    learned_keys = [k for k in ("q_red_final","q_green_final","q_blue_final") if k in df_episodes.columns]
    if learned_keys:
        plt.figure(figsize=(8,4))
        melted = df_episodes.melt(id_vars=["_condition"], value_vars=learned_keys, var_name="value", value_name="learned")
        sns.boxplot(x="_condition", y="learned", hue="value", data=melted, showfliers=False)
        plt.title("Learned values (final)")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "learned_values_box.png", dpi=200)
        plt.close()
    # Save a concise console summary
    print("\nSummary saved to:", OUTPUT_DIR.resolve())
    print(" - combined_per_condition.csv")
    print(" - inferential_summary.csv")
    if len(all_posthoc): print(" - posthoc_results.csv")
    print(" - boxplot PNGs for analyzed metrics")
    print("\nDone.")

if __name__ == "__main__":
    main()
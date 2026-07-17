
import argparse
from pathlib import Path
import copy
import json

from experiment_config import ExperimentConfig
from lesions import LesionProfile
from generate_dataset_runner import run as run_experiment
from generate_dataset_runner import _lesion_profile_for_name

LESION_TYPES = ["pfc", "parietal", "basal_ganglia", "cerebellum"]

# Severity multipliers: 1.0 = default (as in _lesion_profile_for_name), lower values milder, higher more severe.
SEVERITIES = {
    "mild": 0.75,
    "moderate": 0.50,
    "severe": 0.25
}

def main():
    parser = argparse.ArgumentParser(description="Run a small lesion severity sweep for Experiment 2.")
    parser.add_argument("--config", "-c", type=str, default="exp2_lesion.yml", help="Base YAML config (experiment 2)")
    parser.add_argument("--outdir", "-o", type=str, default="results/lesion_sweep", help="Output directory for CSVs")
    parser.add_argument("--repeats", "-r", type=int, default=5, help="Independent repeats per condition (different seeds)")
    parser.add_argument("--seed_start", type=int, default=2000, help="First seed used; increments per repeat")
    args = parser.parse_args()

    outdir = Path(r"C:\Users\sadyk\Downloads\neuromorphic-control-project-main\neuromorphic-control-project-main\results")
    outdir.mkdir(parents=True, exist_ok=True)

    # Load base config
    cfg = ExperimentConfig.load_from_yaml(args.config)
    base_cfg = cfg

    run_index = 0
    for lesion in LESION_TYPES:
        for severity_name, severity_scale in SEVERITIES.items():
            for rep in range(args.repeats):
                cfg = copy.deepcopy(base_cfg)
                cfg.seed = int(args.seed_start + run_index)
                cfg.lesion = lesion 
                cfg.output_csv = str(outdir / f"results_{lesion}_{severity_name}_seed{cfg.seed}.csv")
                print()
                print("=" * 80)
                print(f"Run {run_index}: lesion={lesion} severity={severity_name} seed={cfg.seed}")
                print(f"Output -> {cfg.output_csv}")
                print("=" * 80)
                # Build explicit lesion_profile_map: only the targeted module gets the generated profile
                lesion_map = {k: None for k in ["vision", "prefrontal", "parietal", "basal_ganglia", "cerebellum"]}
                lesion_map_name = lesion
                # Map "basal_ganglia" if user provided "bg" shorthand, ensure consistent naming
                lesion_map_key = lesion_map_name
                # Use helper to produce a base LesionProfile for this lesion & severity
                lp = _lesion_profile_for_name(lesion_map_name, severity_scale=severity_scale)
                # Assign to the right key in the map: pfc -> prefrontal, parietal -> parietal, basal_ganglia -> basal_ganglia, cerebellum -> cerebellum
                if lesion_map_name == "pfc":
                    lesion_map["prefrontal"] = lp
                elif lesion_map_name in ("basal_ganglia", "bg"):
                    lesion_map["basal_ganglia"] = lp
                elif lesion_map_name == "parietal":
                    lesion_map["parietal"] = lp
                elif lesion_map_name in ("cerebellum", "cb"):
                    lesion_map["cerebellum"] = lp
                elif lesion_map_name == "vision":
                    lesion_map["vision"] = lp

                run_experiment(cfg, lesion_profile_map=lesion_map, severity_scale=severity_scale)

                meta = {
                    "run_index": run_index,
                    "lesion": lesion,
                    "severity": severity_name,
                    "severity_scale": severity_scale,
                    "seed": cfg.seed,
                    "output_csv": cfg.output_csv,
                }
                with open(outdir / f"run_{run_index:04d}.json", "w", encoding="utf-8") as fh:
                    json.dump(meta, fh, indent=2)
                run_index += 1

    print("\nLesion sweep complete. Results written to:", outdir.resolve())


if __name__ == "__main__":
    main()
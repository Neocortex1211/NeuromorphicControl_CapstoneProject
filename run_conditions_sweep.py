# run_conditions_sweep.py
import argparse
from pathlib import Path
from prefrontal import Context
from experiment_config import ExperimentConfig
from generate_dataset_runner import run
import time

CONDITION_MAP = {
    "reward": Context.REWARD_SEEKING,
    "energy": Context.ENERGY_EFFICIENT,
    "explore": Context.EXPLORATION
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="exp1_behavioral.yml", help="Base YAML config")
    parser.add_argument("--repeats", "-r", type=int, default=20, help="Independent repeats per condition (different seeds)")
    parser.add_argument("--seed_start", type=int, default=1000, help="First seed used; increments per repeat")
    args = parser.parse_args()

    try:
        base_cfg = ExperimentConfig.load_from_yaml(args.config)
    except FileNotFoundError as e:
        print("Config file not found:", e)
        return

    print("Using config:", args.config)
    print("cfg.episodes =", base_cfg.episodes, "cfg.steps_per_episode =", base_cfg.steps_per_episode)
    print("Repeats per condition:", args.repeats)

    out_dir = Path(r"C:\Users\sadyk\Downloads\neuromorphic-control-project-main\neuromorphic-control-project-main\results\codition_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    cond_names = list(CONDITION_MAP.keys())

    for cond_idx, (cond_name, ctx) in enumerate(CONDITION_MAP.items()):
        for rep in range(args.repeats):
            cfg = ExperimentConfig.load_from_yaml(args.config)  # fresh copy
            cfg.experiment = 1
            cfg.fixed_context = ctx
            seed = int(args.seed_start + cond_idx * args.repeats + rep)
            cfg.seed = seed
            cfg.output_csv = str(out_dir / f"results_{cond_name}_seed{seed}.csv")
            print()
            print("="*60)
            print(f"Running condition={cond_name} seed={seed} -> {cfg.output_csv}")
            print("="*60)
            run(cfg)

    print("\nSweep finished. Results are in:", out_dir.resolve())

if __name__ == "__main__":
    main()
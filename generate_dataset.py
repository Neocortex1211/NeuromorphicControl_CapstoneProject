
from pathlib import Path
import argparse

from experiment_config import ExperimentConfig
from generate_dataset_runner import run as run_experiment

def main():
    parser = argparse.ArgumentParser(description="Generate dataset using ExperimentConfig YAML or defaults.")
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to YAML config (optional). If omitted ExperimentConfig() defaults are used.")
    args = parser.parse_args()

    if args.config:
        cfg = ExperimentConfig.load_from_yaml(args.config)
    else:
        cfg = ExperimentConfig()

    run_experiment(cfg)

if __name__ == "__main__":
    main()
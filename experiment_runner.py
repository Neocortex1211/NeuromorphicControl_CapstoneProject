import argparse
from experiment_config import ExperimentConfig
from generate_dataset_runner import run

def main():
    parser = argparse.ArgumentParser(description="Run neuromorphic control experiment from YAML config.")
    parser.add_argument("--config", "-c", type=str, default="exp1_behavioral.yaml", help="Path to experiment YAML config")
    args = parser.parse_args()

    cfg = ExperimentConfig.load_from_yaml(args.config)
    cfg.print_configuration()
    run(cfg)

if __name__ == "__main__":
    main()
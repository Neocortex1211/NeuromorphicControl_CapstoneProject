
import argparse
from pathlib import Path
import copy
import json

from experiment_config import ExperimentConfig
from generate_dataset_runner import run as run_experiment

DEFAULT_MOTIONS = ["static", "sin", "circle", "random"]
DEFAULT_SPEEDS = [0.5, 1.0, 2.0]        # cycles per second
DEFAULT_RADII = [0.06, 0.12, 0.18]      # meters

def main():
    parser = argparse.ArgumentParser(description="Run dynamic target sweeps for Experiment 3.")
    parser.add_argument("--config", "-c", type=str, default="exp3_dynamic.yml", help="Base YAML config")
    parser.add_argument("--outdir", "-o", type=str, default="results/dynamic_sweep", help="Output folder")
    parser.add_argument("--repeats", "-r", type=int, default=6, help="Independent repeats per condition")
    parser.add_argument("--seed_start", type=int, default=6000, help="First seed used; increments per run")
    parser.add_argument("--episodes", type=int, default=None, help="Override cfg.episodes for sweep")
    parser.add_argument("--steps", type=int, default=None, help="Override cfg.steps_per_episode for sweep")
    parser.add_argument("--motions", nargs="+", default=None, help="Motion types to sweep (static,sin,circle,random)")
    parser.add_argument("--speeds", nargs="+", type=float, default=None, help="Speeds (cycles/sec)")
    parser.add_argument("--radii", nargs="+", type=float, default=None, help="Target radii (meters)")
    args = parser.parse_args()

    outdir = Path(r"C:\Users\sadyk\Downloads\neuromorphic-control-project-main\neuromorphic-control-project-main\results\dynamic_sweep")
    outdir.mkdir(parents=True, exist_ok=True)

    base_cfg = ExperimentConfig.load_from_yaml(args.config)

    motions = args.motions if args.motions is not None else DEFAULT_MOTIONS
    speeds = args.speeds if args.speeds is not None else DEFAULT_SPEEDS
    radii = args.radii if args.radii is not None else DEFAULT_RADII

    run_idx = 0
    for motion in motions:
        for speed in speeds:
            for radius in radii:
                for rep in range(args.repeats):
                    cfg = copy.deepcopy(base_cfg)
                    cfg.dynamic_targets = True
                    cfg.target_motion = motion
                    cfg.target_speed = float(speed)
                    cfg.target_radius = float(radius)
                    if args.episodes is not None:
                        cfg.episodes = int(args.episodes)
                    if cfg.episodes is None:
                        cfg.episodes = 20
                    if args.steps is not None:
                        cfg.steps_per_episode = int(args.steps)
                    if cfg.steps_per_episode is None:
                        cfg.steps_per_episode = 1000

                    cfg.seed = int(args.seed_start + run_idx)
                    cfg.output_csv = str(outdir / f"results_{motion}_sp{int(speed*100)}_r{int(radius*100)}_seed{cfg.seed}.csv")

                    print("="*80)
                    print(f"Run {run_idx}: motion={motion} speed={speed} radius={radius} seed={cfg.seed}")
                    print("Episodes:", cfg.episodes, "Steps:", cfg.steps_per_episode)
                    print("Output:", cfg.output_csv)
                    print("="*80)

                    run_experiment(cfg)

                    meta = {"idx": run_idx, "motion": motion, "speed": speed, "radius": radius, "seed": cfg.seed, "out": cfg.output_csv}
                    with open(outdir / f"run_{run_idx:04d}.json", "w", encoding="utf-8") as fh:
                        json.dump(meta, fh, indent=2)

                    run_idx += 1

    print("Dynamic sweep finished. Results in", outdir.resolve())

if __name__ == "__main__":
    main()
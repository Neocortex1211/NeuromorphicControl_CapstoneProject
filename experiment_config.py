
from dataclasses import dataclass
from prefrontal import Context
from typing import Any, Dict, Optional
from pathlib import Path
import yaml
import os


_DEFAULT_FILENAMES = [
    "exp1_lesion.yml",
    "exp1_lesion.yaml",
    "exp1_behavioral.yml",
    "exp1_behavioral.yaml",
    "exp1.yml",
    "exp1.yaml",
    "config.yml",
    "config.yaml",
]


def _candidate_paths(user_path: Optional[str]) -> list:
    """
    Given a user-supplied path (or None), return an ordered list of Path
    candidates to try for locating the YAML file.
    """
    candidates = []
    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent

    def add_variants(base):
        p = Path(base)
        if p.suffix in (".yml", ".yaml"):
            candidates.append(p)
        else:
            candidates.append(p.with_suffix(".yml"))
            candidates.append(p.with_suffix(".yaml"))
            candidates.append(p)  # keep original too

    if user_path:
        user = Path(user_path)
        # 1) exact as given (absolute or relative)
        candidates.append(user)
        # 2) cwd / user
        candidates.append(cwd / user)
        # 3) script dir / user
        candidates.append(script_dir / user)
        # 4) cwd / config / user
        candidates.append(cwd / "config" / user)
        # 5) script_dir / config / user
        candidates.append(script_dir / "config" / user)
        # 6) try basename variants (if user gave only a name)
        add_variants(user)
    else:
        for name in _DEFAULT_FILENAMES:
            candidates.append(Path(name))
            candidates.append(cwd / name)
            candidates.append(script_dir / name)
            candidates.append(script_dir / "config" / name)
            candidates.append(cwd / "config" / name)

    
    cur = cwd
    for _ in range(5):
        candidates.append(cur / "config" / (user_path or _DEFAULT_FILENAMES[0]))
        cur = cur.parent

    # remove duplicates while preserving order
    seen = set()
    unique = []
    for p in candidates:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if str(rp) not in seen:
            seen.add(str(rp))
            unique.append(p)
    return unique


@dataclass
class ExperimentConfig:
    # General
    seed: int = 42
    episodes: int = 50
    steps_per_episode: int = 1000
    xml_file: str = "armscene.xml"
    output_csv: str = "results.csv"

    # Experiment (1=Behavioural Context, 2=Lesion Study, 3=Dynamic Targets)
    experiment: int = 1

    # Demonstration
    demo_mode: bool = False

    # Behavioural Context (Experiment 1)
    fixed_context: Context = Context.REWARD_SEEKING

    # Lesion condition for Experiment 2
    lesion: str = "none"

    randomize_initial_pose: bool = True

    # Vision
    sensor_noise_std: float = 0.0
    delayed_vision: bool = False
    vision_delay: float = 0.05       # seconds of visual delay when delayed_vision=True
    occlusion_probability: float = 0.0

    # Environment (Experiment 3)
    dynamic_targets: bool = False
    moving_after_episode: int = 9999
    target_motion: str = "sin"
    target_speed: float = 0.00
    target_radius: float = 0.00
    random_walk_std: float = 0.002
    moving_distractor: bool = False  # if True, only one distractor target moves

    # External perturbations
    random_force: bool = False
    force_std: float = 0.0

    # Reaching task
    movement_timeout: float = 2.0
    endpoint_threshold: float = 0.03

    # Logging
    save_video: bool = False
    verbose: bool = True

    @property
    def experiment_name(self) -> str:
        names = {1: "Behavioural Context", 2: "Brain Lesion", 3: "Dynamic Target Tracking"}
        return names.get(self.experiment, "Unknown")

    def print_configuration(self) -> None:
        print()
        print("=" * 60)
        print("Experiment Configuration")
        print("=" * 60)
        print(f"Experiment : {self.experiment_name}")
        print(f"Context    : {self.fixed_context.name}")
        print(f"Lesion     : {self.lesion}")
        print(f"XML file   : {self.xml_file}")
        print(f"CSV Output : {self.output_csv}")
        print(f"Dynamic    : {self.dynamic_targets}")
        print(f"Seed       : {self.seed}")
        print("=" * 60)

    def update_from_dict(self, data: Dict[str, Any]) -> None:
        for k, v in data.items():
            if not hasattr(self, k):
                continue
            if k == "fixed_context":
                if isinstance(v, str):
                    try:
                        self.fixed_context = Context[v.strip()]
                        continue
                    except Exception:
                        for c in Context:
                            if c.name.lower() == v.strip().lower():
                                self.fixed_context = c
                                break
                        continue
                elif isinstance(v, (int, float)):
                    try:
                        self.fixed_context = Context(int(v))
                    except Exception:
                        pass
                    continue
            # standard assignment
            setattr(self, k, v)

    @classmethod
    def load_from_yaml(cls, path: Optional[str] = None) -> "ExperimentConfig":
        """
        Load ExperimentConfig from a YAML file. The path argument can be:
        - an absolute path
        - a path relative to the current working directory
        - a filename with/without extension (the loader tries .yml and .yaml)
        - None (in which case default filenames and locations are tried)

        If file is not found a FileNotFoundError is raised showing attempted locations.
        """
        tried = []
        for candidate in _candidate_paths(path):
            tried.append(str(candidate))
            try_paths = []
            if candidate.suffix in (".yml", ".yaml"):
                try_paths.append(candidate)
            else:
                try_paths.append(candidate.with_suffix(".yml"))
                try_paths.append(candidate.with_suffix(".yaml"))
            try_paths.append(candidate)
            for p in try_paths:
                try:
                    p_resolved = p.resolve()
                except Exception:
                    p_resolved = p
                if p_resolved.exists():
                    with open(p_resolved, "r", encoding="utf-8") as fh:
                        data = yaml.safe_load(fh) or {}
                    cfg = cls()
                    cfg.update_from_dict(data)
                    return cfg
        msg = (
            "ExperimentConfig.load_from_yaml: could not find YAML file.\n"
            "Tried the following locations (in order):\n"
            + "\n".join(tried)
            + "\n\nPass --config <path> with an absolute path or place the file in a 'config' folder next to the script or in the current working directory."
        )
        raise FileNotFoundError(msg)
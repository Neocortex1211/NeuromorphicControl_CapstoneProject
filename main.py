
import argparse
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

from experiment_config import ExperimentConfig
from utils import create_module_rngs
from lesions import LesionProfile
from environment import Environment
from brain_state import BrainState
from vision import Vision
from prefrontal import PrefrontalCortex
from parietal import ParietalCortex
from basal_ganglia import BasalGanglia
from motor_cortex import MotorCortex
from trajectory_generator import TrajectoryGenerator
from inverse_kinematics import InverseKinematics
from spinal_cord import SpinalCord
from cerebellum import Cerebellum
from logger import Logger


def instantiate_modules(cfg, rngs):
    """
    Create modules and return as dict. Vision receives cfg so it can implement
    delayed vision / occlusion and compute target velocities.
    """
    vision = Vision(
        model,
        rng=rngs["vision"],
        lesion=(LesionProfile(noise_std=cfg.sensor_noise_std) if cfg.sensor_noise_std > 0 else None),
        cfg=cfg
    )

    prefrontal = PrefrontalCortex(
        rng=rngs["prefrontal"],
        fixed_context=cfg.fixed_context,
        lesion=(LesionProfile(gain_scale=0.6, learning_scale=0.6) if cfg.lesion == "pfc" else None)
    )

    parietal = ParietalCortex(
        rng=rngs["parietal"],
        lesion=(LesionProfile(noise_std=0.20, gain_scale=0.7) if cfg.lesion == "parietal" else None)
    )

    basal_ganglia = BasalGanglia(
        rng=rngs["basal_ganglia"],
        lesion=(LesionProfile(gain_scale=0.45, noise_std=2.0, learning_scale=0.15, temperature_mult=1.5) if cfg.lesion == "basal_ganglia" else None)
    )

    motor_cortex = MotorCortex(demo_mode=cfg.demo_mode)

    trajectory = TrajectoryGenerator()

    ik = InverseKinematics()

    spinal = SpinalCord(demo_mode=cfg.demo_mode)

    cerebellum = Cerebellum(
        rng=rngs["cerebellum"],
        lesion=(LesionProfile(gain_scale=0.15, noise_std=0.8, learning_scale=0.2) if cfg.lesion == "cerebellum" else None)
    )

    return {
        "vision": vision,
        "prefrontal": prefrontal,
        "parietal": parietal,
        "basal_ganglia": basal_ganglia,
        "motor": motor_cortex,
        "trajectory": trajectory,
        "ik": ik,
        "spinal": spinal,
        "cerebellum": cerebellum
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the neuromorphic control viewer.")
    parser.add_argument("--config", "-c", type=str, default="exp3_dynamic.yml", help="Path to YAML config")
    parser.add_argument("--record", action="store_true", help="Record logged data to CSV while viewing")
    args = parser.parse_args()

    cfg = ExperimentConfig.load_from_yaml(args.config)
    cfg.print_configuration()

    HERE = Path(__file__).parent

    # Create per-module RNGs
    module_names = [
        "environment", "vision", "prefrontal", "parietal", "basal_ganglia",
        "motor_cortex", "trajectory", "inverse_kinematics", "spinal_cord",
        "cerebellum", "viewer"
    ]
    rngs, seeds = create_module_rngs(cfg.seed, module_names)

    # Load model
    model_path = HERE / cfg.xml_file
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo XML not found: {model_path.resolve()}")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    hand_site = model.site("hand").id

    # Environment and modules
    environment = Environment(model, cfg, rng=rngs["environment"])
    modules = instantiate_modules(cfg, rngs)

    # Optional logger
    logger = Logger() if args.record else None
    if logger is not None:
        logger.set_metadata({
            "root_seed": int(cfg.seed),
            "module_seeds": seeds,
            "config": {
                "file": str(args.config),
                "experiment": cfg.experiment,
                "lesion": cfg.lesion,
                "fixed_context": str(cfg.fixed_context),
                "dynamic_targets": cfg.dynamic_targets,
                "target_motion": cfg.target_motion,
                "target_speed": cfg.target_speed,
                "target_radius": cfg.target_radius
            }
        })

    # Reset and forward once
    environment.reset(data)
    mujoco.mj_forward(model, data)

    # Shared brain state
    state = BrainState()
    state.lesion = cfg.lesion
    state.endpoint_threshold = cfg.endpoint_threshold
    state.movement_timeout = cfg.movement_timeout

    # Viewer loop
    with mujoco.viewer.launch_passive(model, data) as viewer:
        try:
            while viewer.is_running():
                # Environment
                environment.update(data, episode=cfg.moving_after_episode)
                mujoco.mj_forward(model, data)

                # Perception & processing
                modules["vision"].perceive(data, state)
                modules["prefrontal"].process(state, data.time)
                modules["parietal"].process(state)
                modules["basal_ganglia"].process(state, data.time)

                if state.selected_plan is not None:
                    modules["motor"].process(state, data.time)

                    # target velocity if available (lead)
                    t_vel = None
                    if hasattr(state.selected_plan.target, "velocity"):
                        t_vel = state.selected_plan.target.velocity

                    modules["trajectory"].set_goal(
                        goal_position=state.goal_position,
                        current_position=state.hand_position,
                        current_time=data.time,
                        movement_vigor=state.movement_vigor,
                        target_velocity=t_vel
                    )

                    modules["trajectory"].process(state, data.time)
                    modules["ik"].process(state, data)
                    modules["spinal"].process(state, data)
                    modules["cerebellum"].process(state, data)
                    data.ctrl[:] = state.corrected_torque

                mujoco.mj_step(model, data)
                viewer.sync()

                # Optional logging per-step
                if logger is not None:
                    hand = data.site_xpos[hand_site][:2].copy()
                    goal = state.goal_position.copy() if state.selected_plan is not None else np.zeros(2)
                    logger.record(
                        time=data.time,
                        target=(state.selected_plan.target.name if state.selected_plan is not None else ""),
                        endpoint_error=float(np.linalg.norm(hand - goal)),
                        tracking_error=float(state.tracking_error),
                        q1=float(data.qpos[0]), q2=float(data.qpos[1]),
                        tau1=float(state.corrected_torque[0]), tau2=float(state.corrected_torque[1]),
                        goal_bias=state.goal_bias, learning_gain=state.learning_gain,
                    )
        except Exception as e:
            print("Viewer loop terminated with error:", e)
            raise

    # Save logger if used
    if logger is not None:
        out = HERE / cfg.output_csv
        logger.save(out)
        print("Saved recording to:", out)
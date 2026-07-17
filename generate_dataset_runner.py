
from pathlib import Path
import numpy as np
import mujoco
import dataclasses
from typing import Optional, Dict

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

from environment import Environment
from logger import Logger
from lesions import LesionProfile
from utils import create_module_rngs


def _lesion_profile_for_name(name: str, severity_scale: float = 1.0) -> Optional[LesionProfile]:
    """
    Return a reasonable LesionProfile for a named lesion. The severity_scale
    multiplies sensible fields (gain_scale and noise_std) to produce mild->severe variants.
    """
    if name is None:
        return None

    n = name.lower()
    if n == "pfc":
        return LesionProfile(
            gain_scale=max(0.0, 0.6 * severity_scale),
            noise_std=0.0,
            learning_scale=max(0.0, 0.6 * severity_scale),
            temperature_mult=1.25
        )
    if n == "parietal":
        return LesionProfile(
            gain_scale=max(0.0, 0.8 * severity_scale),
            noise_std=0.08 * severity_scale,
            learning_scale=1.0,
            temperature_mult=1.0
        )
    if n in ("basal_ganglia", "bg"):
        return LesionProfile(
            gain_scale=max(0.0, 0.5 * severity_scale),
            noise_std=1.0 * severity_scale,
            learning_scale=max(0.0, 0.4 * severity_scale),
            temperature_mult=2.0
        )
    if n in ("cerebellum", "cb"):
        return LesionProfile(
            gain_scale=max(0.0, 0.2 * severity_scale),
            noise_std=0.8 * severity_scale,
            learning_scale=max(0.0, 0.3 * severity_scale),
            temperature_mult=1.0
        )
    if n == "vision":
        return LesionProfile(
            gain_scale=1.0,
            noise_std=0.02 * severity_scale,
            learning_scale=1.0,
            temperature_mult=1.0
        )
    return None


def _build_default_lesion_map(cfg, severity_scale: float = 1.0) -> Dict[str, Optional[LesionProfile]]:
    """
    Build a default lesion map from cfg.lesion string.
    Only the module named in cfg.lesion gets a non-None LesionProfile; others remain None.
    """
    modules = ["vision", "prefrontal", "parietal", "basal_ganglia", "cerebellum"]
    lesion_map = {}
    for m in modules:
        if cfg.lesion and cfg.lesion.lower() in (m, "bg", "cb", "pfc"):
            lesion_map[m] = _lesion_profile_for_name(cfg.lesion, severity_scale=severity_scale)
        else:
            lesion_map[m] = None
    return lesion_map


def run(cfg, lesion_profile_map: Optional[Dict[str, Optional[LesionProfile]]] = None, severity_scale: float = 1.0):
    """
    Run dataset generation for a given ExperimentConfig.

    Parameters
    ----------
    cfg : ExperimentConfig
    lesion_profile_map : dict or None  (module_name -> LesionProfile)
    severity_scale : float
        multiplicative scale used when constructing default lesion mapping
    """
    HERE = Path(__file__).parent

    # Create stable per-module RNGs
    module_names = [
        "environment", "vision", "prefrontal", "parietal", "basal_ganglia",
        "motor_cortex", "trajectory", "inverse_kinematics", "spinal_cord",
        "cerebellum", "logger"
    ]
    rngs, seeds = create_module_rngs(cfg.seed, module_names)

    model = mujoco.MjModel.from_xml_path(str(HERE / cfg.xml_file))
    hand_site = model.site("hand").id

    environment = Environment(model, cfg, rng=rngs["environment"])
    logger = Logger()

    # Determine lesion_profile_map and metadata
    if lesion_profile_map is None:
        lesion_profile_map = _build_default_lesion_map(cfg, severity_scale=severity_scale)

    # serializable map for metadata
    serializable_map = {}
    for k, v in lesion_profile_map.items():
        serializable_map[k] = dataclasses.asdict(v) if v is not None else None

    logger.set_metadata({
        "root_seed": int(cfg.seed),
        "module_seeds": seeds,
        "lesion_profile_map": serializable_map,
        "experiment_config": {
            "experiment": cfg.experiment,
            "lesion": cfg.lesion,
            "fixed_context": str(cfg.fixed_context),
            "dynamic_targets": cfg.dynamic_targets,
            "target_motion": cfg.target_motion,
            "target_speed": cfg.target_speed,
            "target_radius": cfg.target_radius
        },
    })

    # find forearm body id for applying external force (if available)
    try:
        forearm_body_id = model.body("forearm").id
    except Exception:
        forearm_body_id = None

    for episode in range(cfg.episodes):
        print()
        print("=" * 70)
        print(f"Episode {episode+1}/{cfg.episodes}")
        print("=" * 70)

        data = mujoco.MjData(model)
        environment.reset(data)

        if cfg.randomize_initial_pose:
            env_rng = rngs["environment"]
            data.qpos[0] = env_rng.uniform(-0.7, 0.7)
            data.qpos[1] = env_rng.uniform(0.2, 1.8)
            data.qvel[0] = env_rng.uniform(-0.2, 0.2)
            data.qvel[1] = env_rng.uniform(-0.2, 0.2)

        mujoco.mj_forward(model, data)

        # Shared brain state
        state = BrainState()
        state.lesion = cfg.lesion
        state.endpoint_threshold = cfg.endpoint_threshold
        state.movement_timeout = cfg.movement_timeout

        # Instantiate modules with their RNGs and lesion profiles
        vision = Vision(model, rng=rngs["vision"], lesion=lesion_profile_map.get("vision"), cfg=cfg)
        prefrontal = PrefrontalCortex(rng=rngs["prefrontal"], fixed_context=cfg.fixed_context, lesion=lesion_profile_map.get("prefrontal"))
        parietal = ParietalCortex(rng=rngs["parietal"], lesion=lesion_profile_map.get("parietal"))
        basal_ganglia = BasalGanglia(rng=rngs["basal_ganglia"], lesion=lesion_profile_map.get("basal_ganglia"))
        motor = MotorCortex(demo_mode=cfg.demo_mode)
        trajectory = TrajectoryGenerator()
        ik = InverseKinematics()
        spinal = SpinalCord(demo_mode=cfg.demo_mode)
        cerebellum = Cerebellum(rng=rngs["cerebellum"], lesion=lesion_profile_map.get("cerebellum"))

        for step in range(cfg.steps_per_episode):
            state.episode = episode
            state.step = step
            state.simulation_time = data.time

            # Environment updates (dynamic targets if enabled)
            environment.update(data, episode)

            # apply external force to forearm if configured
            if cfg.random_force and forearm_body_id is not None:
                f = environment.external_force()  # XY force
                try:
                    data.xfrc_applied[forearm_body_id, :3] = 0.0
                    data.xfrc_applied[forearm_body_id, :3] += np.array([f[0], f[1], 0.0])
                except Exception:
                    pass

            mujoco.mj_forward(model, data)

            # Processing chain
            vision.perceive(data, state)
            prefrontal.process(state, data.time)
            parietal.process(state)
            basal_ganglia.process(state, data.time)

            if state.selected_plan is not None:
                motor.process(state, data.time)
                t_vel = None
                if state.selected_plan is not None and hasattr(state.selected_plan.target, "velocity"):
                     t_vel = state.selected_plan.target.velocity
                     trajectory.set_goal(goal_position=state.goal_position, current_position=state.hand_position, current_time=data.time, movement_vigor=state.movement_vigor, target_velocity=t_vel)
                trajectory.process(state, data.time)
                ik.process(state, data)
                spinal.process(state, data)
                cerebellum.process(state, data)
                data.ctrl[:] = state.corrected_torque

            mujoco.mj_step(model, data)

            # Measurements and logging
            hand = data.site_xpos[hand_site][:2].copy()
            goal = state.goal_position.copy() if state.selected_plan is not None else np.zeros(2)
            state.endpoint_error = float(np.linalg.norm(hand - goal))
            if state.tracking_error <= 0.0:
                state.tracking_error = float(np.linalg.norm(hand - state.desired_position))

            rewards = {t.name: t.reward for t in state.targets}
            probabilities = state.action_probabilities.copy()
            values = state.learned_values.copy()

            logger.record(
                experiment=cfg.experiment,
                experiment_name=cfg.experiment_name,
                lesion=cfg.lesion,
                context=state.context.name,
                episode=episode,
                step=step,
                time=data.time,
                target=(state.selected_plan.target.name if state.selected_plan is not None else ""),
                reward=(state.selected_plan.target.reward if state.selected_plan is not None else np.nan),
                utility=(state.selected_plan.utility if state.selected_plan is not None else np.nan),
                probability=state.selected_probability,
                decision_confidence=state.decision_confidence,
                decision_entropy=state.decision_entropy,
                dopamine=state.dopamine_error,
                expected_reward=state.expected_reward,
                movement_vigor=state.movement_vigor,
                dynamic_targets=cfg.dynamic_targets,
                target_motion=cfg.target_motion,
                reward_red=rewards.get("Red", np.nan),
                reward_green=rewards.get("Green", np.nan),
                reward_blue=rewards.get("Blue", np.nan),
                q_red=values.get("Red", np.nan),
                q_green=values.get("Green", np.nan),
                q_blue=values.get("Blue", np.nan),
                p_red=probabilities.get("Red", np.nan),
                p_green=probabilities.get("Green", np.nan),
                p_blue=probabilities.get("Blue", np.nan),
                hand_x=hand[0],
                hand_y=hand[1],
                goal_x=goal[0],
                goal_y=goal[1],
                endpoint_error=state.endpoint_error,
                tracking_error=state.tracking_error,
                trajectory_progress=state.trajectory_progress,
                correction_magnitude=state.correction_magnitude,
                q1=data.qpos[0],
                q2=data.qpos[1],
                dq1=data.qvel[0],
                dq2=data.qvel[1],
                desired_q1=state.desired_joint_angles[0],
                desired_q2=state.desired_joint_angles[1],
                tau1=state.corrected_torque[0],
                tau2=state.corrected_torque[1],
                ff_tau1=state.feedforward_torque[0],
                ff_tau2=state.feedforward_torque[1],
                goal_bias=state.goal_bias,
                learning_gain=state.learning_gain,
                exploration_gain=state.exploration_gain,
                motor_gain=state.motor_gain,
                attention_gain=state.attention_gain,
            )

    dataset = logger.save(HERE / cfg.output_csv)

    print()
    print("=" * 70)
    print("Dataset generation completed")
    print("=" * 70)
    print(f"Experiment : {cfg.experiment_name}")
    print(f"Lesion     : {cfg.lesion}")
    print(f"Episodes   : {cfg.episodes}")
    print(f"Samples    : {len(dataset)}")
    print(f"Saved to   : {HERE / cfg.output_csv}")
    print("=" * 70)

    return dataset
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np

@dataclass
class Target:
    name: str
    position: np.ndarray
    reward: float
    obstacle_cost: float = 0.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))


@dataclass
class MotorPlan:
    target: Target
    # Geometry
    distance: float
    direction: np.ndarray = field(default_factory=lambda: np.zeros(2))
    unit_direction: np.ndarray = field(default_factory=lambda: np.zeros(2))
    # Parietal representation
    salience: float = 0.0
    certainty: float = 1.0
    movement_cost: float = 0.0
    # Basal ganglia
    utility: float = 0.0
    probability: float = 0.0

@dataclass
class BrainState:
    # ======================================================
    # Perception
    # ======================================================
    hand_position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    hand_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    targets: List[Target] = field(default_factory=list)

    # ======================================================
    # Executive state (PFC)
    # ======================================================
    context: Any = None
    goal_bias: float = 1.0
    learning_gain: float = 1.0
    exploration_gain: float = 1.0
    motor_gain: float = 1.0
    attention_gain: float = 1.0

    # ======================================================
    # Spatial representation (Parietal)
    # ======================================================
    candidate_plans: List[MotorPlan] = field(default_factory=list)

    # ======================================================
    # Decision (Basal Ganglia)
    # ======================================================
    selected_plan: Optional[MotorPlan] = None
    action_probabilities: Dict[str, float] = field(default_factory=dict)
    selected_probability: float = 0.0
    decision_entropy: float = 0.0
    decision_confidence: float = 0.0
    movement_vigor: float = 1.0
    expected_reward: float = 0.0

    # ======================================================
    # Reinforcement learning
    # ======================================================
    dopamine_error: float = 0.0
    learned_values: Dict[str, float] = field(default_factory=lambda: {
        "Red": 7.0,
        "Green": 7.0,
        "Blue": 7.0
    })

    # ======================================================
    # Movement state
    # ======================================================
    movement_active: bool = False
    movement_finished: bool = False
    movement_start_time: float = 0.0
    movement_duration: float = 0.0
    movement_timeout: float = 3.0
    endpoint_threshold: float = 0.07
    current_target_name: str = ""

    # ======================================================
    # Motor Cortex outputs (task-space)
    # ======================================================
    goal_position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    desired_position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    desired_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    desired_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(2))
    desired_joint_angles: np.ndarray = field(default_factory=lambda: np.zeros(2))
    desired_q1: float = 0.0
    desired_q2: float = 0.0

    # ======================================================
    # Joint-space control and torques
    # ======================================================
    desired_torque: np.ndarray = field(default_factory=lambda: np.zeros(2))
    feedforward_torque: np.ndarray = field(default_factory=lambda: np.zeros(2))
    corrected_torque: np.ndarray = field(default_factory=lambda: np.zeros(2))

    # ======================================================
    # Cerebellar outputs
    # ======================================================
    predicted_endpoint: np.ndarray = field(default_factory=lambda: np.zeros(2))
    correction_magnitude: float = 0.0
    tracking_error: float = 0.0

    # ======================================================
    # Performance
    # ======================================================
    desired_force: np.ndarray = field(default_factory=lambda: np.zeros(2))
    corrected_force: np.ndarray = field(default_factory=lambda: np.zeros(2))
    position_error: np.ndarray = field(default_factory=lambda: np.zeros(2))
    endpoint_error: float = 0.0

    # ======================================================
    # Additional diagnostics / bookkeeping
    # ======================================================
    trajectory_progress: float = 0.0
    reachable_distance: float = 0.0

    episode: int = 0
    step: int = 0
    simulation_time: float = 0.0
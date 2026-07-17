import numpy as np


class TrajectoryGenerator:
    def __init__(
        self,
        base_speed=0.80,
        min_duration=0.25,
        max_duration=1.20,
    ):
        self.base_speed = base_speed
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.active = False
        self.start_time = 0.0
        self.duration = 0.80
        self.start_position = np.zeros(2)
        self.goal_position = np.zeros(2)

    def set_goal(
        self,
        goal_position,
        current_position,
        current_time,
        movement_vigor=1.0,
        target_velocity=None,
    ):
        goal_position = goal_position.copy()

        # -----------------------------------------
        # Apply a small predictive lead if target_velocity provided
        # -----------------------------------------
        if target_velocity is not None:
            # compute nominal duration (distance / speed) to estimate lead time
            nominal_distance = np.linalg.norm(goal_position - current_position)
            vigor = np.clip(movement_vigor, 0.5, 1.5)
            nominal_speed = self.base_speed * vigor
            nominal_duration = nominal_distance / max(nominal_speed, 1e-6)
            nominal_duration = np.clip(nominal_duration, self.min_duration, self.max_duration)
            lead_time = float(np.clip(0.25 * nominal_duration, 0.05, 0.6))
            goal_position = goal_position + lead_time * target_velocity


        if (not self.active) or np.linalg.norm(goal_position - self.goal_position) > 1e-4:
            self.start_position = current_position.copy()
            self.goal_position = goal_position
            self.start_time = current_time
            distance = np.linalg.norm(self.goal_position - self.start_position)

            vigor = np.clip(movement_vigor, 0.5, 1.5)
            speed = self.base_speed * vigor
            duration = distance / max(speed, 1e-6)
            duration = np.clip(duration, self.min_duration, self.max_duration)
            self.duration = float(duration)
            self.active = True

    def process(self, state, current_time):
        if not self.active:
            state.desired_position = state.goal_position.copy()
            state.desired_velocity = np.zeros(2)
            state.desired_acceleration = np.zeros(2)
            return

        tau = (current_time - self.start_time) / self.duration
        tau = np.clip(tau, 0.0, 1.0)

        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / self.duration
        dds = (60 * tau - 180 * tau**2 + 120 * tau**3) / (self.duration**2)

        delta = self.goal_position - self.start_position
        desired_position = self.start_position + s * delta
        desired_velocity = ds * delta
        desired_acceleration = dds * delta

        state.desired_position = desired_position
        state.desired_velocity = desired_velocity
        state.desired_acceleration = desired_acceleration

        state.trajectory_progress = tau

        if tau >= 1.0:
            self.active = False
            state.desired_position = self.goal_position.copy()
            state.desired_velocity = np.zeros(2)
            state.desired_acceleration = np.zeros(2)
            state.trajectory_progress = 1.0
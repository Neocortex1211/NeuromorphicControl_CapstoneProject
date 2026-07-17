import numpy as np


class SpinalCord:
    """
    Spinal Cord

    Joint-space PD controller.
    """

    def __init__(self, demo_mode=False):
        self.demo_mode = demo_mode
        if demo_mode:
            self.Kp = np.array([320.0, 320.0])
            self.Kd = 2.0 * np.sqrt(self.Kp)
            self.max_torque = np.array([300.0, 300.0])
            self.torque_alpha = 0.45
        else:
            self.Kp = np.array([220.0, 220.0])
            self.Kd = 2.0 * np.sqrt(self.Kp)
            self.max_torque = np.array([180.0, 180.0])
            self.torque_alpha = 0.25

        self.previous_torque = np.zeros(2)

    def process(self, state, data):
        q = data.qpos[:2].copy()
        dq = data.qvel[:2].copy()
        q_des = state.desired_joint_angles.copy()
        dq_des = np.zeros(2)

        position_error = q_des - q
        velocity_error = dq_des - dq

        torque = (self.Kp * position_error) + (self.Kd * velocity_error)

        if hasattr(state, "feedforward_torque"):
            torque += state.feedforward_torque

        torque = np.clip(torque, -self.max_torque, self.max_torque)

        torque = (self.torque_alpha * torque) + ((1.0 - self.torque_alpha) * self.previous_torque)
        self.previous_torque = torque.copy()

        state.position_error = position_error
        state.velocity_error = velocity_error
        state.desired_torque = torque.copy()
        state.corrected_torque = torque.copy()
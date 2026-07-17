import numpy as np


class InverseKinematics:
    def __init__(self):
        self.L1 = 0.40
        self.L2 = 0.35

    def process(self, state, data=None):
        x = float(state.desired_position[0])
        y = float(state.desired_position[1])

        r = np.hypot(x, y)
        max_r = self.L1 + self.L2 - 1e-6
        min_r = abs(self.L1 - self.L2) + 1e-6

        if r > max_r:
            scale = max_r / r
            x *= scale
            y *= scale
        elif 0 < r < min_r:
            scale = min_r / r
            x *= scale
            y *= scale

        state.ik_target = np.array([x, y])

        c2 = (x*x + y*y - self.L1*self.L1 - self.L2*self.L2) / (2*self.L1*self.L2)
        c2 = np.clip(c2, -1.0, 1.0)
        s2 = np.sqrt(max(0.0, 1.0 - c2*c2))
        q2_up = np.arctan2(s2, c2)
        q2_down = np.arctan2(-s2, c2)

        def shoulder(q2):
            k1 = self.L1 + self.L2*np.cos(q2)
            k2 = self.L2*np.sin(q2)
            return np.arctan2(y, x) - np.arctan2(k2, k1)

        q1_up = shoulder(q2_up)
        q1_down = shoulder(q2_down)

        solution_up = np.array([q1_up, q2_up])
        solution_down = np.array([q1_down, q2_down])

        if data is not None:
            current = data.qpos[:2].copy()
            d_up = np.linalg.norm(solution_up - current)
            d_down = np.linalg.norm(solution_down - current)
            if d_up <= d_down:
                chosen = solution_up
                state.ik_solution = "elbow_up"
            else:
                chosen = solution_down
                state.ik_solution = "elbow_down"
        else:
            chosen = solution_up
            state.ik_solution = "elbow_up"

        q1 = np.arctan2(np.sin(chosen[0]), np.cos(chosen[0]))
        q2 = np.arctan2(np.sin(chosen[1]), np.cos(chosen[1]))

        state.desired_joint_angles = np.array([q1, q2])
        state.desired_q1 = q1
        state.desired_q2 = q2
        state.reachable_distance = np.hypot(x, y)
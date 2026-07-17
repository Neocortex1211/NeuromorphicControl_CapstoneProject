import numpy as np


class MotorCortex:

    def __init__(self, demo_mode=False, lesion=False):
        self.demo_mode = demo_mode
        self.lesion = lesion

        if demo_mode:
            self.base_alpha = 0.55
            self.base_speed = 3.5
        else:
            self.base_alpha = 0.25
            self.base_speed = 1.5

        self.previous_desired = None
        self.last_second = -1

    def process(self, state, sim_time):
        if state.selected_plan is None:
            return

        # Current state
        hand = state.hand_position.copy()
        goal = state.selected_plan.target.position.copy()
        state.goal_position = goal.copy()

        if self.previous_desired is None:
            self.previous_desired = hand.copy()

        # Inputs from previous modules
        confidence = max(0.10, state.decision_confidence)
        vigor = max(0.20, state.movement_vigor)
        certainty = max(0.20, state.selected_plan.certainty)

        if self.lesion:
            confidence *= 0.4
            vigor *= 0.5
            certainty *= 0.5

        distance = np.linalg.norm(goal - hand)
        alpha = (self.base_alpha + 0.50 * confidence * certainty)
        alpha = np.clip(alpha, 0.15, 0.95)

        desired = ((1.0 - alpha) * self.previous_desired) + (alpha * goal)
        self.previous_desired = desired.copy()

        # Desired velocity (task space)
        desired_velocity = desired - hand
        speed = np.linalg.norm(desired_velocity)
        max_speed = (self.base_speed * vigor)
        if speed > max_speed and speed > 0:
            desired_velocity *= (max_speed / speed)

        desired_acceleration = np.zeros(2)

        # Tracking error (task-space)
        tracking_error = np.linalg.norm(goal - desired)

        state.desired_position = desired
        state.desired_velocity = desired_velocity
        state.desired_acceleration = desired_acceleration
        state.feedforward_torque = np.zeros(2)
        state.tracking_error = tracking_error

        second = int(sim_time)
        if second != self.last_second:
            self.last_second = second
            print()
            print("=" * 70)
            print("Motor Cortex")
            print("=" * 70)
            print(f"Time                 : {sim_time:.2f}")
            print(f"Target               : {state.selected_plan.target.name}")
            print(f"Goal                 : {np.round(goal,3)}")
            print(f"Hand                 : {np.round(hand,3)}")
            print()
            print(f"Decision confidence  : {confidence:.3f}")
            print(f"Movement vigor       : {vigor:.3f}")
            print(f"Spatial certainty    : {certainty:.3f}")
            print()
            print(f"Smoothing alpha      : {alpha:.3f}")
            print(f"Desired speed        : {np.linalg.norm(desired_velocity):.3f}")
            print(f"Tracking error       : {tracking_error:.3f}")
            print("=" * 70)
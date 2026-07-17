# parietal.py
import numpy as np
from brain_state import MotorPlan
from lesions import LesionProfile

class ParietalCortex:
    def __init__(self, rng: np.random.Generator, lesion: LesionProfile = None):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.lesion = lesion

    def process(self, state):
        state.candidate_plans.clear()
        hand = state.hand_position
        attention = state.attention_gain
        for target in state.targets:
            displacement = target.position - hand
            distance = np.linalg.norm(displacement)
            if distance > 1e-8:
                direction = displacement
                unit_direction = displacement / distance
            else:
                direction = np.zeros(2)
                unit_direction = np.zeros(2)
            if self.lesion is not None:
                noisy_distance = max(0.0, distance + self.rng.normal(0.20, 0.10))
                certainty = np.clip(self.rng.normal(0.45, 0.15), 0.10, 0.80)
                if self.lesion.noise_std > 0:
                    noisy_distance = max(0.0, distance + self.rng.normal(0.0, self.lesion.noise_std))
                    certainty = np.clip(0.5 * (1.0 - self.lesion.gain_scale), 0.05, 0.9) if self.lesion.gain_scale < 1.0 else certainty
            else:
                noisy_distance = distance
                certainty = np.clip(0.95 * attention, 0.0, 1.5)
            salience = (attention * target.reward) / (1.0 + noisy_distance)
            movement_cost = noisy_distance + target.obstacle_cost
            plan = MotorPlan(target=target, distance=noisy_distance)
            plan.direction = direction
            plan.unit_direction = unit_direction
            plan.salience = salience
            plan.certainty = certainty
            plan.movement_cost = movement_cost
            state.candidate_plans.append(plan)
        if self.lesion is not None:
            print("\n" + "=" * 50)
            print("Parietal Cortex (LESION)")
            print("=" * 50)
            for plan in state.candidate_plans:
                print(f"{plan.target.name:6s} Dist={plan.distance:5.2f} Cert={plan.certainty:4.2f} Salience={plan.salience:5.2f}")
            print("=" * 50)
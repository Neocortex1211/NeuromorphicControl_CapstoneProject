# basal_ganglia.py
import numpy as np
from prefrontal import Context
from lesions import LesionProfile

class BasalGanglia:

    def __init__(self, rng: np.random.Generator, lesion: LesionProfile = None):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.lesion = lesion

        self.values = {
            "Red": 0.0,
            "Green": 0.0,
            "Blue": 0.0
        }

        # Learning and commitment
        self.learning_rate = 0.18  # slightly higher to see changes within episodes
        self.commitment_time = 0.20  # seconds -- shorter commitment means multiple choices per episode
        self.last_decision_time = -np.inf

    # -------------------------------------------------------------

    def context_policy(self, context):
        if context == Context.REWARD_SEEKING:
            return {"reward": 2.2, "distance": 0.7, "salience": 1.6, "certainty": 0.8, "cost": 0.4}
        elif context == Context.ENERGY_EFFICIENT:
            return {"reward": 1.0, "distance": 2.0, "salience": 0.8, "certainty": 0.8, "cost": 1.6}
        elif context == Context.EXPLORATION:
            # Exploration increases the weight on salience and allows higher temperature via PFC
            return {"reward": 1.2, "distance": 1.0, "salience": 1.0, "certainty": 0.8, "cost": 0.8}
        return {"reward": 1.0, "distance": 1.0, "salience": 1.0, "certainty": 1.0, "cost": 1.0}

    # -------------------------------------------------------------

    def process(self, state, current_time):
        # Action commitment gating
        if current_time - self.last_decision_time < self.commitment_time:
            return

        weights = self.context_policy(state.context)
        utilities = []

        for plan in state.candidate_plans:
            expected_value = self.values.get(plan.target.name, 0.0)
            utility = (
                state.goal_bias * weights["reward"] * expected_value
                + weights["salience"] * plan.salience
                + weights["certainty"] * plan.certainty
                - weights["distance"] * plan.distance
                - weights["cost"] * plan.movement_cost
            )

            # Lesion: reduces discrimination and add noise
            if self.lesion is not None:
                utility *= self.lesion.gain_scale
                if self.lesion.noise_std > 0.0:
                    utility += self.rng.normal(0.0, self.lesion.noise_std)

            plan.utility = float(utility)
            utilities.append(utility)

        if len(utilities) == 0:
            return

        utilities = np.asarray(utilities)
        # Normalize utilities for numerical stability
        utilities = (utilities - utilities.mean()) / (utilities.std() + 1e-8)

        # Temperature determined by exploration gain from PFC; lesion can inflate it
        temperature = float(max(0.01, state.exploration_gain))
        if self.lesion is not None:
            temperature = max(0.01, temperature * self.lesion.temperature_mult)

        probs = np.exp(utilities / temperature)
        probs = probs / probs.sum()

        chosen_idx = int(self.rng.choice(len(state.candidate_plans), p=probs))
        chosen_plan = state.candidate_plans[chosen_idx]

        state.selected_plan = chosen_plan
        state.selected_probability = float(probs[chosen_idx])
        state.action_probabilities = {p.target.name: float(pr) for p, pr in zip(state.candidate_plans, probs)}

        # Decision stats
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        state.decision_entropy = float(entropy)
        state.decision_confidence = float(1.0 - entropy / np.log2(len(probs)))

        state.expected_reward = float(self.values.get(chosen_plan.target.name, 0.0))
        state.movement_vigor = float(state.motor_gain * state.decision_confidence)

        # Dopamine learning (TD-like single-step update)
        reward = float(chosen_plan.target.reward)
        prediction = float(self.values.get(chosen_plan.target.name, 0.0))
        delta = reward - prediction

        lr = float(self.learning_rate * state.learning_gain)
        if self.lesion is not None:
            lr *= self.lesion.learning_scale

        self.values[chosen_plan.target.name] = self.values.get(chosen_plan.target.name, 0.0) + lr * delta

        state.learned_values = self.values.copy()
        state.dopamine_error = float(delta)

        # Commit and book-keep
        self.last_decision_time = current_time
        state.current_target_name = chosen_plan.target.name
        state.movement_start_time = current_time
        state.movement_active = True
        state.movement_finished = False

        print()
        print("=" * 70)
        print("Basal Ganglia")
        print("=" * 70)
        print("Context:", state.context.name)
        for plan in state.candidate_plans:
            p_pct = 100.0 * state.action_probabilities.get(plan.target.name, 0.0)
            print(f"{plan.target.name:6s}  U={plan.utility:7.2f}  Sal={plan.salience:5.2f}  Cert={plan.certainty:5.2f}  Dist={plan.distance:5.2f}  P={p_pct:5.1f}%")
        print()
        print("Chosen:", chosen_plan.target.name)
        print(f"Decision confidence : {state.decision_confidence:.3f}")
        print(f"Movement vigor      : {state.movement_vigor:.3f}")
        print(f"Dopamine δ          : {delta:+.3f}")
        print(f"Learning rate       : {lr:.3f}")
        print("=" * 70)
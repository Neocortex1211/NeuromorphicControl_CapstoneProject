# prefrontal.py
from enum import Enum
from lesions import LesionProfile
import numpy as np

class Context(Enum):
    REWARD_SEEKING = 0
    ENERGY_EFFICIENT = 1
    EXPLORATION = 2

class PrefrontalCortex:
    """
    Executive controller: provides executive signals (goal_bias, learning_gain, exploration_gain, motor_gain, attention_gain)
    """

    def __init__(self, rng: np.random.Generator, fixed_context=None, switch_interval=12.0, lesion: LesionProfile = None):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.fixed_context = fixed_context
        self.switch_interval = switch_interval
        self.lesion = lesion
        self.contexts = [Context.REWARD_SEEKING, Context.ENERGY_EFFICIENT, Context.EXPLORATION]
        self.current_index = 0
        self.last_switch = 0.0

    def current(self):
        if self.fixed_context is not None:
            return self.fixed_context
        return self.contexts[self.current_index]

    def executive_signals(self):
        if self.lesion is not None:
            return {
                "goal_bias": 1.0 * self.lesion.gain_scale,
                "learning_gain": 0.60 * self.lesion.learning_scale,
                "exploration_gain": 1.25 * self.lesion.temperature_mult,
                "motor_gain": 0.80 * self.lesion.gain_scale,
                "attention_gain": 0.60 * self.lesion.gain_scale,
            }

        context = self.current()

        if context == Context.REWARD_SEEKING:
            return {"goal_bias": 1.30, "learning_gain": 1.15, "exploration_gain": 0.75, "motor_gain": 1.10, "attention_gain": 1.20}
        elif context == Context.ENERGY_EFFICIENT:
            return {"goal_bias": 0.90, "learning_gain": 0.90, "exploration_gain": 0.80, "motor_gain": 0.80, "attention_gain": 0.95}
        elif context == Context.EXPLORATION:
            # stronger exploration gain to induce higher policy entropy
            return {"goal_bias": 1.00, "learning_gain": 1.00, "exploration_gain": 2.50, "motor_gain": 1.00, "attention_gain": 1.00}

        return {"goal_bias": 1.0, "learning_gain": 1.0, "exploration_gain": 1.0, "motor_gain": 1.0, "attention_gain": 1.0}

    def process(self, state, current_time):
        if self.fixed_context is None:
            if current_time - self.last_switch >= self.switch_interval:
                self.current_index = (self.current_index + 1) % len(self.contexts)
                self.last_switch = current_time
                print()
                print("=" * 40)
                print("Prefrontal Cortex")
                print("Context ->", self.current().name)
                print("=" * 40)

        state.context = self.current()
        signals = self.executive_signals()
        state.goal_bias = signals["goal_bias"]
        state.learning_gain = signals["learning_gain"]
        state.exploration_gain = signals["exploration_gain"]
        state.motor_gain = signals["motor_gain"]
        state.attention_gain = signals["attention_gain"]

        if self.lesion is not None:
            print()
            print("=" * 40)
            print("Prefrontal Cortex (LESION)")
            print("=" * 40)
            print("Executive control degraded")
            print(f"Context            : {state.context.name}")
            print(f"Goal bias          : {state.goal_bias:.2f}")
            print(f"Learning gain      : {state.learning_gain:.2f}")
            print(f"Attention gain     : {state.attention_gain:.2f}")
            print(f"Exploration gain   : {state.exploration_gain:.2f}")
            print(f"Motor gain         : {state.motor_gain:.2f}")
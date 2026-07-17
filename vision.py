# vision.py
import numpy as np
from brain_state import Target
from lesions import LesionProfile
from typing import Optional

class Vision:
    """
    Visual Cortex — deterministic RNG & lesion profile
    Now computes approximate target velocities (finite difference) and supports delayed vision / occlusion.
    """
    def __init__(self, model, rng: np.random.Generator, lesion: LesionProfile = None, cfg: Optional[object] = None):
        self.model = model
        self.rng = rng if rng is not None else np.random.default_rng()
        self.lesion = lesion
        self.cfg = cfg

        self.hand_id = model.site("hand").id
        self.red_id = model.site("target_red").id
        self.green_id = model.site("target_green").id
        self.blue_id = model.site("target_blue").id

        # Base sensor noise
        self.sensor_noise_std = 0.0

        # For finite-difference velocity estimates
        self.prev_positions = {}
        self.prev_time = None

        # Delayed vision
        self.delay_buffer = []
        self.max_buffer_len = 2000
        self.delay_seconds = getattr(cfg, "vision_delay", 0.05) if cfg is not None else 0.05
        self.use_delay = getattr(cfg, "delayed_vision", False) if cfg is not None else False
        self.occlusion_prob = getattr(cfg, "occlusion_probability", 0.0) if cfg is not None else 0.0

    # ---------------------------------------------------------
    def observe(self, position):
        p = position.copy()
        std = self.sensor_noise_std
        if self.lesion is not None and self.lesion.noise_std > std:
            std = max(std, self.lesion.noise_std)
        if std > 0.0:
            p += self.rng.normal(0.0, std, 2)
        return p

    # ---------------------------------------------------------
    def _append_buffer(self, t, hand, red, green, blue):
        self.delay_buffer.append((t, hand.copy(), red.copy(), green.copy(), blue.copy()))
        if len(self.delay_buffer) > self.max_buffer_len:
            self.delay_buffer.pop(0)

    def _get_delayed_snapshot(self, t):
        if not self.delay_buffer:
            return None
        target_time = t - self.delay_seconds
        chosen = None
        for ts, hand, red, green, blue in self.delay_buffer:
            if ts <= target_time:
                chosen = (ts, hand, red, green, blue)
            else:
                break
        if chosen is None:
            return self.delay_buffer[0]
        return chosen

    # ---------------------------------------------------------
    def reward_schedule(self, sim_time):
        phase = int(sim_time // 5) % 3
        if phase == 0:
            rewards = {"Red": 10.0, "Green": 7.0, "Blue": 4.0}
        elif phase == 1:
            rewards = {"Red": 4.0, "Green": 10.0, "Blue": 7.0}
        else:
            rewards = {"Red": 7.0, "Green": 4.0, "Blue": 10.0}
        for k in rewards:
            rewards[k] += self.rng.normal(0.0, 0.20)
        return rewards

    # ---------------------------------------------------------
    def perceive(self, data, state):
        # Raw noisy observations
        hand_pos_raw = self.observe(data.site_xpos[self.hand_id][:2])
        red_pos_raw = self.observe(data.site_xpos[self.red_id][:2])
        green_pos_raw = self.observe(data.site_xpos[self.green_id][:2])
        blue_pos_raw = self.observe(data.site_xpos[self.blue_id][:2])

        self._append_buffer(data.time, hand_pos_raw, red_pos_raw, green_pos_raw, blue_pos_raw)

        if self.use_delay:
            snap = self._get_delayed_snapshot(data.time)
            if snap is None:
                _, hand_pos, red_pos, green_pos, blue_pos = self.delay_buffer[-1]
            else:
                _, hand_pos, red_pos, green_pos, blue_pos = snap
        else:
            hand_pos, red_pos, green_pos, blue_pos = hand_pos_raw, red_pos_raw, green_pos_raw, blue_pos_raw

        # occlusion
        def maybe_occlude(pos):
            if self.rng.random() < self.occlusion_prob:
                return None
            return pos

        red_pos = maybe_occlude(red_pos)
        green_pos = maybe_occlude(green_pos)
        blue_pos = maybe_occlude(blue_pos)

        # compute velocities via finite differences using prev_positions
        dt = None
        if self.prev_time is not None:
            dt = data.time - self.prev_time
            if dt <= 1e-8:
                dt = None

        def compute_velocity(name, pos):
            if pos is None:
                return np.zeros(2)
            if dt is None or name not in self.prev_positions:
                return np.zeros(2)
            prev = self.prev_positions[name]
            return (pos - prev) / dt

        if red_pos is not None:
            self.prev_positions["Red"] = red_pos.copy()
        if green_pos is not None:
            self.prev_positions["Green"] = green_pos.copy()
        if blue_pos is not None:
            self.prev_positions["Blue"] = blue_pos.copy()
        self.prev_time = data.time

        # builds Target objects with velocity
        rewards = self.reward_schedule(data.time)

        targets = []
        if red_pos is not None:
            v = compute_velocity("Red", red_pos)
            targets.append(Target(name="Red", position=red_pos, reward=rewards["Red"], velocity=v))
        if green_pos is not None:
            v = compute_velocity("Green", green_pos)
            targets.append(Target(name="Green", position=green_pos, reward=rewards["Green"], velocity=v))
        if blue_pos is not None:
            v = compute_velocity("Blue", blue_pos)
            targets.append(Target(name="Blue", position=blue_pos, reward=rewards["Blue"], velocity=v))

        state.hand_position = hand_pos
        state.targets = targets
        state.simulation_time = data.time
        state.visible_targets = len(targets)
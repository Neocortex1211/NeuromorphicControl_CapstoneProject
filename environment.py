# environment.py
import numpy as np

class Environment:

    def __init__(self, model, config, rng: np.random.Generator):
        self.model = model
        self.cfg = config
        self.rng = rng if rng is not None else np.random.default_rng()

        self.targets = {
            "Red": (model.joint("red_x").qposadr[0], model.joint("red_y").qposadr[0]),
            "Green": (model.joint("green_x").qposadr[0], model.joint("green_y").qposadr[0]),
            "Blue": (model.joint("blue_x").qposadr[0], model.joint("blue_y").qposadr[0]),
        }

        self.phase = {"Red": 0.0, "Green": 2*np.pi/3, "Blue": 4*np.pi/3}

        self.initial_offsets = {name: np.zeros(2) for name in self.targets}

        self.previous_offsets = {name: np.zeros(2) for name in self.targets}
        self.target_velocity = {name: np.zeros(2) for name in self.targets}
        self.previous_time = 0.0

        self.distractor_name = None
        if getattr(self.cfg, "moving_distractor", False):
            # pick one target deterministically using rng
            self.distractor_name = self.rng.choice(list(self.targets.keys()))


    def reset(self, data):
        for name, (qx, qy) in self.targets.items():
            # record current qpos as initial offset
            self.initial_offsets[name][0] = data.qpos[qx]
            self.initial_offsets[name][1] = data.qpos[qy]
            data.qpos[qx] = self.initial_offsets[name][0]
            data.qpos[qy] = self.initial_offsets[name][1]
        self.previous_time = data.time
        for name in self.targets:
            self.previous_offsets[name][:] = self.initial_offsets[name].copy()
            self.target_velocity[name][:] = 0.0


    def update(self, data, episode):
        if not getattr(self.cfg, "dynamic_targets", False):
            return

        if episode < getattr(self.cfg, "moving_after_episode", 0):
            return

        motion = getattr(self.cfg, "target_motion", "sin").lower()

        if getattr(self.cfg, "moving_distractor", False) and self.distractor_name is not None:
            names_to_move = [self.distractor_name]
        else:
            names_to_move = list(self.targets.keys())

        if motion == "sin":
            self._sinusoidal(data, names_to_move)
        elif motion == "circle":
            self._circular(data, names_to_move)
        elif motion == "random":
            self._random_walk(data, names_to_move)
        elif motion == "static":
            for name, (qx, qy) in self.targets.items():
                data.qpos[qx] = self.initial_offsets[name][0]
                data.qpos[qy] = self.initial_offsets[name][1]
        else:
            pass

        self._update_velocities(data)


    def _sinusoidal(self, data, names_to_move):
        t = data.time
        amplitude = getattr(self.cfg, "target_radius", 0.0)
        omega = 2 * np.pi * max(0.0, getattr(self.cfg, "target_speed", 0.0))
        for name in names_to_move:
            qx, qy = self.targets[name]
            data.qpos[qx] = self.initial_offsets[name][0]
            data.qpos[qy] = self.initial_offsets[name][1] + amplitude * np.sin(omega * t + self.phase[name])


    def _circular(self, data, names_to_move):
        t = data.time
        amplitude = getattr(self.cfg, "target_radius", 0.0)
        omega = 2 * np.pi * max(0.0, getattr(self.cfg, "target_speed", 0.0))
        for name in names_to_move:
            qx, qy = self.targets[name]
            angle = omega * t + self.phase[name]
            data.qpos[qx] = self.initial_offsets[name][0] + amplitude * np.cos(angle)
            data.qpos[qy] = self.initial_offsets[name][1] + amplitude * np.sin(angle)

    # =========================================================

    def _random_walk(self, data, names_to_move):
        sigma = getattr(self.cfg, "random_walk_std", 0.002)
        limit = 0.20
        for name in names_to_move:
            qx, qy = self.targets[name]
            data.qpos[qx] += self.rng.normal(0.0, sigma)
            data.qpos[qy] += self.rng.normal(0.0, sigma)
            # bound relative to initial offsets so target stays near workspace
            data.qpos[qx] = np.clip(data.qpos[qx], self.initial_offsets[name][0] - limit, self.initial_offsets[name][0] + limit)
            data.qpos[qy] = np.clip(data.qpos[qy], self.initial_offsets[name][1] - limit, self.initial_offsets[name][1] + limit)

    # =========================================================

    def _update_velocities(self, data):
        dt = data.time - self.previous_time
        if dt <= 1e-8:
            return
        for name, (qx, qy) in self.targets.items():
            current = np.array([data.qpos[qx], data.qpos[qy]])
            velocity = (current - self.previous_offsets[name]) / dt
            self.target_velocity[name] = velocity
            self.previous_offsets[name] = current.copy()
        self.previous_time = data.time

    # =========================================================

    def get_target_velocity(self, name):
        return self.target_velocity.get(name, np.zeros(2)).copy()

    # =========================================================

    def external_force(self):
        """
        External perturbations.

        Returns a 2D random force (x,y) if random_force is enabled in cfg, else zeros.
        """
        if not getattr(self.cfg, "random_force", False):
            return np.zeros(2)
        return self.rng.normal(0.0, getattr(self.cfg, "force_std", 0.0), 2)
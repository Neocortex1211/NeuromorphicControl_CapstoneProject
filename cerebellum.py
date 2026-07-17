from pathlib import Path
import numpy as np
import torch
from neural_model import CerebellumNet
from lesions import LesionProfile


class Cerebellum:

    def __init__(self, rng, lesion: LesionProfile = None):
        self.rng = rng
        self.lesion = lesion
        self.device = torch.device("cpu")
        self.network = CerebellumNet().to(self.device)
        model_path = Path(__file__).parent / "models" / "cerebellum.pth"
        self.enabled = False
        if model_path.exists():
            self.network.load_state_dict(torch.load(model_path, map_location=self.device))
            self.enabled = True
            print("Loaded trained cerebellum.")
        else:
            print("No trained cerebellum found. Using spinal controller only.")
        self.network.eval()

    def process(self, state, data):
        q = data.qpos[:2].copy()
        dq = data.qvel[:2].copy()
        q_des = state.desired_joint_angles.copy()
        base_torque = state.desired_torque.copy()

        correction = np.zeros(2)
        confidence = 0.0

        if self.enabled:
            x = np.array([q[0], q[1], dq[0], dq[1], q_des[0], q_des[1]], dtype=np.float32)
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                correction = self.network(x).cpu().numpy().flatten()
            confidence = 1.0

        if self.lesion is not None:
            correction *= max(0.0, self.lesion.gain_scale)
            confidence *= max(0.0, self.lesion.gain_scale)
            if self.lesion.noise_std > 0.0:
                correction += self.rng.normal(0.0, self.lesion.noise_std, size=2)

        correction *= 0.25
        corrected_torque = base_torque + correction

        correction_norm = np.linalg.norm(correction)
        predicted_endpoint = state.desired_position + 0.02 * correction

        state.corrected_torque = corrected_torque
        state.feedforward_torque = correction
        state.predicted_endpoint = predicted_endpoint
        state.correction_magnitude = float(correction_norm)
        state.tracking_error = float(np.linalg.norm(state.desired_position - state.hand_position) + 0.5 * correction_norm)

        if self.lesion is not None:
            print("\n" + "=" * 60)
            print("Cerebellum (LESION)")
            print("=" * 60)
            print(f"Prediction confidence : {confidence:.2f}")
            print(f"Correction magnitude  : {correction_norm:.3f}")
            print(f"Tracking error        : {state.tracking_error:.3f}")
            print("=" * 60)
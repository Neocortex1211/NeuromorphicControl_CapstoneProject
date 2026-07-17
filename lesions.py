# lesions.py
from dataclasses import dataclass

@dataclass
class LesionProfile:
    gain_scale: float = 1.0
    noise_std: float = 0.0
    learning_scale: float = 1.0
    temperature_mult: float = 1.0
    delay_ms: float = 0.0
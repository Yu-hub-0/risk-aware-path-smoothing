"""Configurable realism disturbances for the quadrotor simulation."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DisturbanceConfig:
    """Independent switches and magnitudes for simulation disturbances."""

    model_mismatch: bool = False
    wind: bool = False
    measurement_noise: bool = False
    control_delay: bool = False
    initial_state_offset: bool = False
    motor_limits: bool = False

    # The controller keeps the nominal parameters; these scale the real plant.
    actual_mass_scale: float = 1.10
    actual_inertia_scale: float = 1.15
    actual_drag_scale: float = 1.20

    constant_wind_accel: np.ndarray = field(
        default_factory=lambda: np.array([0.25, -0.20, 0.05], dtype=float))
    gust_std: float = 0.08

    position_noise_std: float = 0.02
    velocity_noise_std: float = 0.03
    attitude_noise_std_deg: float = 0.25
    body_rate_noise_std: float = 0.01

    # 20 ms is a moderate unmodelled delay. Around one full 50 ms NMPC period
    # is an aggressive stress test and can destabilize direct wrench control.
    delay_seconds: float = 0.02

    initial_position_offset: np.ndarray = field(
        default_factory=lambda: np.array([0.50, -0.30, 0.20], dtype=float))
    initial_velocity_offset: np.ndarray = field(
        default_factory=lambda: np.array([0.20, -0.10, 0.05], dtype=float))

    arm_length: float = 0.13
    yaw_moment_ratio: float = 0.02
    max_rotor_thrust_ratio: float = 2.0

    def enabled_names(self):
        names = (
            "model_mismatch",
            "wind",
            "measurement_noise",
            "control_delay",
            "initial_state_offset",
            "motor_limits",
        )
        return [name for name in names if getattr(self, name)]


def disturbance_case(name):
    """Return one standard case with only ``name`` enabled."""
    config = DisturbanceConfig()
    if name == "ideal":
        return config
    if not hasattr(config, name) or not isinstance(getattr(config, name), bool):
        raise ValueError(f"Unknown disturbance case: {name}")
    setattr(config, name, True)
    return config


def disturbance_from_switches(switches):
    """Build a configuration from a user-editable boolean dictionary."""
    config = DisturbanceConfig()
    for name, enabled in switches.items():
        if not hasattr(config, name) or not isinstance(getattr(config, name), bool):
            raise ValueError(f"Unknown disturbance switch: {name}")
        setattr(config, name, bool(enabled))
    return config

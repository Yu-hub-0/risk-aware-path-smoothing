"""Online IMM forecast of quadrotor tracking-error trends.

The estimator operates on the reference-frame tracking error

    e_p = p_measured - p_reference
    e_v = v_measured - v_reference

and maintains three linear Gaussian motion hypotheses for that error:

* CV: acceleration quickly vanishes.
* CA: acceleration persists.
* DM: acceleration decays gradually (short manoeuvre/disturbance).

All models share the state [position_error, velocity_error, acceleration_error],
which makes the standard IMM interaction step well defined.  The forecast
returned to NMPC is an *increment beyond constant-velocity propagation of the
current error*.  The nominal nonlinear plant already starts from the measured
position and velocity, so this subtraction avoids counting the current error
twice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IMMErrorForecast:
    """Prediction consumed by NMPC over one rolling horizon."""

    position_increment: np.ndarray
    velocity_increment: np.ndarray
    position_sigma: np.ndarray
    model_probabilities: np.ndarray
    absolute_position_error: np.ndarray
    absolute_velocity_error: np.ndarray


class _ErrorKalmanModel:
    """Nine-state linear filter with configurable acceleration persistence."""

    def __init__(
        self,
        dt: float,
        acceleration_persistence: float,
        process_std: float,
        position_measurement_std: float,
        velocity_measurement_std: float,
    ):
        self.dt = float(dt)
        self.rho = float(acceleration_persistence)
        self.x = np.zeros(9)
        self.P = np.eye(9)
        self.P[6:9, 6:9] *= 4.0
        self.Q = self._build_process_covariance(float(process_std))
        self.H = np.zeros((6, 9))
        self.H[0:3, 0:3] = np.eye(3)
        self.H[3:6, 3:6] = np.eye(3)
        self.R = np.diag(
            np.r_[
                np.full(3, float(position_measurement_std) ** 2),
                np.full(3, float(velocity_measurement_std) ** 2),
            ]
        )

    def transition(self) -> np.ndarray:
        dt = self.dt
        f = np.eye(9)
        f[0:3, 3:6] = dt * np.eye(3)
        f[0:3, 6:9] = 0.5 * dt**2 * np.eye(3)
        f[3:6, 6:9] = dt * np.eye(3)
        f[6:9, 6:9] = self.rho * np.eye(3)
        return f

    def _build_process_covariance(self, process_std: float) -> np.ndarray:
        # White acceleration-derivative noise, applied independently per axis.
        dt = self.dt
        axis_gain = np.array([dt**3 / 6.0, dt**2 / 2.0, dt])
        q_axis = process_std**2 * np.outer(axis_gain, axis_gain)
        q = np.zeros((9, 9))
        for axis in range(3):
            indices = [axis, axis + 3, axis + 6]
            q[np.ix_(indices, indices)] = q_axis
        q += np.eye(9) * 1e-10
        return q

    def predict(self) -> None:
        f = self.transition()
        self.x = f @ self.x
        self.P = f @ self.P @ f.T + self.Q

    def update(self, measurement: np.ndarray) -> float:
        innovation = measurement - self.H @ self.x
        innovation_cov = self.H @ self.P @ self.H.T + self.R
        try:
            solved = np.linalg.solve(innovation_cov, innovation)
            gain = np.linalg.solve(
                innovation_cov, self.H @ self.P
            ).T
        except np.linalg.LinAlgError:
            innovation_cov = innovation_cov + np.eye(6) * 1e-8
            solved = np.linalg.solve(innovation_cov, innovation)
            gain = np.linalg.solve(
                innovation_cov, self.H @ self.P
            ).T
        self.x = self.x + gain @ innovation
        identity = np.eye(9)
        # Joseph form preserves covariance symmetry and positive semidefiniteness.
        correction = identity - gain @ self.H
        self.P = correction @ self.P @ correction.T + gain @ self.R @ gain.T
        self.P = 0.5 * (self.P + self.P.T)

        sign, logdet = np.linalg.slogdet(innovation_cov)
        if sign <= 0:
            return 1e-300
        log_likelihood = -0.5 * (
            6 * np.log(2.0 * np.pi) + logdet + innovation @ solved
        )
        return float(np.exp(np.clip(log_likelihood, -690.0, 50.0)))


class IMMTrackingErrorPredictor:
    """Interacting-multiple-model predictor for online tracking-error trends."""

    MODEL_NAMES = ("CV", "CA", "DM")

    def __init__(
        self,
        dt: float,
        position_measurement_std: float = 0.05,
        velocity_measurement_std: float = 0.10,
        process_stds=(0.15, 0.80, 0.40),
        transition_matrix=None,
        initial_probabilities=(0.50, 0.25, 0.25),
    ):
        self.dt = float(dt)
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if len(process_stds) != 3:
            raise ValueError("process_stds must contain CV, CA, and DM values")

        # rho=0 approximates CV; rho~1 approximates CA; intermediate rho
        # represents a manoeuvre or disturbance that gradually decays.
        persistence = (0.0, 0.98, 0.70)
        self.models = [
            _ErrorKalmanModel(
                self.dt,
                rho,
                q_std,
                position_measurement_std,
                velocity_measurement_std,
            )
            for rho, q_std in zip(persistence, process_stds)
        ]
        self.transition_matrix = np.asarray(
            transition_matrix
            if transition_matrix is not None
            else [
                [0.94, 0.03, 0.03],
                [0.04, 0.92, 0.04],
                [0.04, 0.04, 0.92],
            ],
            dtype=float,
        )
        if self.transition_matrix.shape != (3, 3):
            raise ValueError("transition_matrix must have shape (3, 3)")
        row_sums = self.transition_matrix.sum(axis=1)
        if np.any(row_sums <= 0):
            raise ValueError("transition_matrix rows must have positive sums")
        self.transition_matrix = self.transition_matrix / row_sums[:, None]

        self.probabilities = np.asarray(initial_probabilities, dtype=float)
        if self.probabilities.shape != (3,) or self.probabilities.sum() <= 0:
            raise ValueError("initial_probabilities must contain three weights")
        self.probabilities /= self.probabilities.sum()
        self.initialized = False
        self.last_measurement = np.zeros(6)

    def _interaction(self, models, probabilities):
        predicted_probabilities = self.transition_matrix.T @ probabilities
        predicted_probabilities = np.maximum(predicted_probabilities, 1e-15)
        mixing = (
            probabilities[:, None] * self.transition_matrix
            / predicted_probabilities[None, :]
        )
        old_x = [model.x.copy() for model in models]
        old_p = [model.P.copy() for model in models]
        for destination, model in enumerate(models):
            mixed_x = sum(
                mixing[source, destination] * old_x[source]
                for source in range(3)
            )
            mixed_p = np.zeros((9, 9))
            for source in range(3):
                delta = old_x[source] - mixed_x
                mixed_p += mixing[source, destination] * (
                    old_p[source] + np.outer(delta, delta)
                )
            model.x = mixed_x
            model.P = 0.5 * (mixed_p + mixed_p.T)
        return predicted_probabilities

    def update(self, position_error, velocity_error) -> np.ndarray:
        measurement = np.r_[
            np.asarray(position_error, dtype=float).reshape(3),
            np.asarray(velocity_error, dtype=float).reshape(3),
        ]
        self.last_measurement = measurement

        if not self.initialized:
            for model in self.models:
                model.x[0:6] = measurement
                model.P[0:6, 0:6] = model.R.copy()
            self.initialized = True
            return self.probabilities.copy()

        prior_probabilities = self._interaction(self.models, self.probabilities)
        likelihoods = np.zeros(3)
        for index, model in enumerate(self.models):
            model.predict()
            likelihoods[index] = model.update(measurement)
        posterior = prior_probabilities * likelihoods
        if not np.all(np.isfinite(posterior)) or posterior.sum() <= 1e-300:
            posterior = prior_probabilities
        self.probabilities = posterior / posterior.sum()
        return self.probabilities.copy()

    def forecast(self, horizon: int) -> IMMErrorForecast:
        horizon = int(horizon)
        if horizon < 1:
            raise ValueError("horizon must be at least one")
        if not self.initialized:
            zeros = np.zeros((horizon, 3))
            return IMMErrorForecast(
                zeros.copy(),
                zeros.copy(),
                zeros.copy(),
                self.probabilities.copy(),
                zeros.copy(),
                zeros.copy(),
            )

        states = [model.x.copy() for model in self.models]
        covariances = [model.P.copy() for model in self.models]
        probabilities = self.probabilities.copy()
        position = np.zeros((horizon, 3))
        velocity = np.zeros((horizon, 3))
        position_sigma = np.zeros((horizon, 3))

        # Forecast on local arrays so online filter state is not modified.
        for step in range(horizon):
            predicted_probabilities = self.transition_matrix.T @ probabilities
            predicted_probabilities = np.maximum(predicted_probabilities, 1e-15)
            mixing = (
                probabilities[:, None]
                * self.transition_matrix
                / predicted_probabilities[None, :]
            )
            mixed_states = []
            mixed_covariances = []
            for destination in range(3):
                mixed_x = sum(
                    mixing[source, destination] * states[source]
                    for source in range(3)
                )
                mixed_p = np.zeros((9, 9))
                for source in range(3):
                    delta = states[source] - mixed_x
                    mixed_p += mixing[source, destination] * (
                        covariances[source] + np.outer(delta, delta)
                    )
                mixed_states.append(mixed_x)
                mixed_covariances.append(mixed_p)

            next_states = []
            next_covariances = []
            for model, mixed_x, mixed_p in zip(
                self.models, mixed_states, mixed_covariances
            ):
                f = model.transition()
                next_states.append(f @ mixed_x)
                next_covariances.append(f @ mixed_p @ f.T + model.Q)
            states = next_states
            covariances = next_covariances
            probabilities = predicted_probabilities / predicted_probabilities.sum()

            mean = sum(
                probabilities[index] * states[index] for index in range(3)
            )
            covariance = np.zeros((9, 9))
            for index in range(3):
                delta = states[index] - mean
                covariance += probabilities[index] * (
                    covariances[index] + np.outer(delta, delta)
                )
            position[step] = mean[0:3]
            velocity[step] = mean[3:6]
            position_sigma[step] = np.sqrt(
                np.maximum(np.diag(covariance)[0:3], 0.0)
            )

        current_position_error = self.last_measurement[0:3]
        current_velocity_error = self.last_measurement[3:6]
        times = self.dt * np.arange(1, horizon + 1)
        constant_velocity_position = (
            current_position_error[None, :]
            + times[:, None] * current_velocity_error[None, :]
        )
        constant_velocity_velocity = np.repeat(
            current_velocity_error[None, :], horizon, axis=0
        )
        return IMMErrorForecast(
            position_increment=position - constant_velocity_position,
            velocity_increment=velocity - constant_velocity_velocity,
            position_sigma=position_sigma,
            model_probabilities=self.probabilities.copy(),
            absolute_position_error=position,
            absolute_velocity_error=velocity,
        )


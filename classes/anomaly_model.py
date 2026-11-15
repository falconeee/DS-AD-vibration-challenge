from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import yaml

from .interface import ModelParams, PredictOutput, TimeSeries, Weights

DEFAULT_PARAMS_PATH = Path("hyperparameters/model_hyperparams.yaml")


def load_model_params(path: Path = DEFAULT_PARAMS_PATH) -> ModelParams:
    """Load model hyperparameters from YAML. Falls back to ModelParams defaults if file missing."""
    if not path.exists():
        return ModelParams()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return ModelParams(**data)


class AnomalyModel:
    """Scores how far the sustained vibration level sits above the asset's own baseline.

    ``fit`` learns, per feature, the upper end of normal operation: a high quantile of
    the samples taken while the machine was running. ``predict`` measures the level each
    feature sustained over the trailing horizon of a window and reports the largest
    ratio to that baseline as the window's severity. Severity 1.0 therefore reads as
    "at the top of what this machine normally does", whatever its size or unit.
    """

    def __init__(
        self,
        params_path: Path | None = None,
        params: ModelParams | None = None,
    ):
        self.weights = Weights()
        self.params = params or load_model_params(params_path or DEFAULT_PARAMS_PATH)

    def _featuring(
        self, samples: TimeSeries
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Return time-ordered epoch seconds, running mask and named feature series."""
        ordered = sorted(samples.data, key=lambda p: p.timestamp)

        timestamps = np.array([p.timestamp.timestamp() for p in ordered], dtype=float)
        vel = np.array([(p.vel_x, p.vel_y, p.vel_z) for p in ordered], dtype=float)
        acc = np.array([(p.acc_x, p.acc_y, p.acc_z) for p in ordered], dtype=float)

        # One magnitude per sensor type: velocity carries the low-frequency faults
        # (unbalance, misalignment, looseness), acceleration the high-frequency ones
        # (bearings, gears). The magnitude does not depend on how the sensor is mounted.
        features = {
            "velocity": np.linalg.norm(vel, axis=1),
            "acceleration": np.linalg.norm(acc, axis=1),
        }

        # A stopped machine says nothing about its condition, and a corrupt reading
        # must not reach the statistics.
        uptime = np.array([p.uptime for p in ordered], dtype=bool)
        valid = np.all([np.isfinite(values) for values in features.values()], axis=0)

        return timestamps, uptime & valid, features

    def fit(self, fitting_samples: TimeSeries) -> None:
        timestamps, running, features = self._featuring(fitting_samples)
        if len(timestamps) < 2 or not running.any():
            raise ValueError(
                "Cannot fit without at least two samples, one of them running"
            )

        self.weights = Weights(
            fitted=True,
            baseline={
                name: np.quantile(values[running], self.params.baseline_quantile)
                for name, values in features.items()
            },
            sampling_period_seconds=np.median(np.diff(timestamps)),
        )

    def _sustained_level(self, values: np.ndarray) -> float:
        """Level reached in at least ``window_anomaly_ratio`` of the horizon.

        The required sample count comes from the horizon's nominal length, not from
        the samples present, so isolated spikes and short running bursts cannot carry
        a window on their own. Without enough evidence the level is zero.
        """
        horizon_samples = (
            self.params.horizon_hours * 3600 / self.weights.sampling_period_seconds
        )
        required = max(1, round(self.params.window_anomaly_ratio * horizon_samples))
        if len(values) < required:
            return 0.0

        return float(np.partition(values, -required)[-required])

    def predict(self, samples: TimeSeries) -> PredictOutput:
        if not self.weights.fitted:
            raise RuntimeError("Model not fitted")
        if not samples.data:
            raise ValueError("Cannot predict on empty TimeSeries")

        timestamps, running, features = self._featuring(samples)
        horizon_start = timestamps[-1] - self.params.horizon_hours * 3600
        scored = running & (timestamps > horizon_start)

        severities = {
            name: self._sustained_level(values[scored]) / self.weights.baseline[name]
            for name, values in features.items()
        }
        driver = max(severities, key=severities.get)

        return PredictOutput(
            anomaly_status=severities[driver] >= self.params.severity_threshold,
            timestamp=max(p.timestamp for p in samples.data),
            severity=severities[driver],
            driver=driver,
        )

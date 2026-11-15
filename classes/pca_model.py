from datetime import timedelta
from pathlib import Path
from typing import Dict, Protocol, Tuple

import numpy as np
import pandas as pd
import yaml

from .interface import FEATURE_NAMES, PcaModelParams, PcaWeights, PredictOutput, TimeSeries

DEFAULT_PARAMS_PATH = Path("hyperparameters/pca_hyperparams.yaml")

MAD_TO_SIGMA = 1.4826  # MAD of a Gaussian times this constant equals its sigma
MIN_AMPLITUDE = 1e-6  # guards the logarithm against a zero reading
MIN_FIT_WINDOWS = 30
EPS = 1e-12


def load_pca_params(path: Path = DEFAULT_PARAMS_PATH) -> PcaModelParams:
    """Load PCA model hyperparameters from YAML. Falls back to PcaModelParams defaults if file missing."""
    if not path.exists():
        return PcaModelParams()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return PcaModelParams(**data)


class _Detector(Protocol):
    """Maps standardized window levels to a deviation score and its per-variable breakdown."""

    def fit(self, Z: np.ndarray) -> None: ...

    def score(self, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (scores [n], contributions [n, n_features]), both non-negative."""
        ...


class _PcaDetector:
    """Hotelling T² and SPE merged into one quadratic index.

    T² measures how far a sample moves *along* the normal correlation structure
    (all axes rising together); SPE measures how far it moves *off* that structure
    (one axis rising alone). Each is divided by its own control limit so the sum,
    ``phi = z' Phi z``, treats both kinds of fault on the same scale.
    """

    def __init__(self, params: PcaModelParams):
        self._variance = params.pca_variance
        self._limit_quantile = params.threshold_quantile
        self.mean: np.ndarray | None = None
        self.loadings: np.ndarray | None = None
        self.eigenvalues: np.ndarray | None = None
        self._phi: np.ndarray | None = None

    def fit(self, Z: np.ndarray) -> None:
        n_features = Z.shape[1]
        self.mean = Z.mean(axis=0)
        centered = Z - self.mean

        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered, rowvar=False))
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], EPS)
        eigenvectors = eigenvectors[:, order]

        explained = np.cumsum(eigenvalues) / eigenvalues.sum()
        n_components = int(np.searchsorted(explained, self._variance) + 1)
        # Always leave a residual subspace, otherwise SPE is identically zero.
        n_components = min(max(n_components, 1), n_features - 1)

        self.loadings = eigenvectors[:, :n_components]
        self.eigenvalues = eigenvalues[:n_components]

        principal = self.loadings @ np.diag(1.0 / self.eigenvalues) @ self.loadings.T
        residual = np.eye(n_features) - self.loadings @ self.loadings.T
        t2 = np.einsum("ij,jk,ik->i", centered, principal, centered)
        spe = np.einsum("ij,jk,ik->i", centered, residual, centered)
        t2_limit = max(np.quantile(t2, self._limit_quantile), EPS)
        spe_limit = max(np.quantile(spe, self._limit_quantile), EPS)

        self._phi = principal / t2_limit + residual / spe_limit

    def score(self, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        centered = Z - self.mean
        projected = centered @ self._phi
        scores = np.einsum("ij,ij->i", projected, centered)
        # Reconstruction-based contribution (Alcala & Qin, 2009): how much of the
        # index disappears when variable j alone is moved to its best-fitting value.
        # Unlike plain T² contributions it is never negative and it points at the
        # right variable for a single-sensor fault.
        contributions = projected**2 / np.diag(self._phi)
        return scores, contributions


_DETECTORS = {"pca": _PcaDetector}


class PcaAnomalyModel:
    """Scores a window by how far its vibration *level* sits from the asset baseline.

    Each window is first reduced to one level per channel and only then compared
    with the baseline. Scoring single readings instead and counting how many
    exceed a limit wastes the evidence a window carries: a shift of 3 sigma held
    for two hours is unmistakable, yet every reading in it is individually
    unremarkable against a heavy-tailed sample distribution.
    """

    def __init__(
        self,
        params_path: Path | None = None,
        params: PcaModelParams | None = None,
    ):
        self.weights = PcaWeights()
        self.params = params or load_pca_params(params_path or DEFAULT_PARAMS_PATH)
        self._detector: _Detector | None = None

    def _featuring(self, samples: TimeSeries) -> pd.DataFrame:
        """Log-amplitude of the six channels, for samples taken while running only.

        Readings taken with the machine stopped sit near the sensor noise floor and
        say nothing about its condition, so they are dropped here, the single entry
        point for both ``fit`` and ``predict``. RMS amplitudes are positive and
        faults act on them multiplicatively, hence the logarithm.
        """
        running = [p for p in samples.data if p.uptime]
        amplitudes = np.array(
            [[getattr(p, name) for name in FEATURE_NAMES] for p in running],
            dtype=float,
        ).reshape(-1, len(FEATURE_NAMES))
        return pd.DataFrame(
            np.log(np.maximum(amplitudes, MIN_AMPLITUDE)),
            index=pd.DatetimeIndex([p.timestamp for p in running]),
            columns=list(FEATURE_NAMES),
        ).sort_index()

    @property
    def _level_quantile(self) -> float:
        """Quantile that summarizes a window.

        A level above the median makes the window react when only part of it is
        elevated (``window_anomaly_ratio`` of the samples is enough) while still
        ignoring isolated spikes.
        """
        return 1.0 - self.params.window_anomaly_ratio

    def _baseline_levels(self, features: pd.DataFrame) -> np.ndarray:
        """Level of every window of the fit period, one ending at each sample."""
        levels = features.rolling(
            pd.Timedelta(hours=self.params.window_hours),
            min_periods=self.params.min_running_samples,
        ).quantile(self._level_quantile)
        return levels.dropna().to_numpy()

    def _standardize(self, levels: np.ndarray) -> np.ndarray:
        return (levels - np.asarray(self.weights.center)) / np.asarray(self.weights.scale)

    def fit(self, fitting_samples: TimeSeries) -> None:
        levels = self._baseline_levels(self._featuring(fitting_samples))
        if len(levels) < MIN_FIT_WINDOWS:
            raise ValueError(
                f"Need at least {MIN_FIT_WINDOWS} running windows to fit, got {len(levels)}"
            )

        # Median/MAD instead of mean/std: the fit period is not guaranteed clean.
        center = np.median(levels, axis=0)
        spread = MAD_TO_SIGMA * np.median(np.abs(levels - center), axis=0)
        scale = np.maximum(spread, self.params.min_log_scale)
        self.weights = PcaWeights(center=center.tolist(), scale=scale.tolist())

        Z = self._standardize(levels)
        inliers = Z[np.abs(Z).max(axis=1) <= self.params.fit_outlier_z]
        if len(inliers) < max(len(Z) // 2, MIN_FIT_WINDOWS):
            inliers = Z

        self._detector = _DETECTORS[self.params.detector](self.params)
        self._detector.fit(inliers)

        # Empirical limit: the theoretical F / chi-square limits assume independent
        # Gaussian samples, which autocorrelated vibration levels are not.
        scores, _ = self._detector.score(inliers)
        threshold = self.params.threshold_margin * np.quantile(
            scores, self.params.threshold_quantile
        )
        self.weights = PcaWeights(
            fitted=True,
            center=center.tolist(),
            scale=scale.tolist(),
            threshold=max(float(threshold), EPS),
        )

    @staticmethod
    def _rank_contributions(contributions: np.ndarray) -> Dict[str, float]:
        """Share of the deviation owed to each variable, largest first."""
        shares = contributions / max(contributions.sum(), EPS)
        ranked = sorted(zip(FEATURE_NAMES, shares), key=lambda item: -item[1])
        return {name: float(share) for name, share in ranked}

    def _scored_span(self, samples: TimeSeries) -> TimeSeries:
        """The trailing ``window_hours`` of a pipeline window.

        The baseline is learned on windows of that length, so a window is scored on
        the same span. The pipeline window itself can then be as long as needed to
        bridge telemetry gaps without changing what the model looks at.
        """
        start = samples.data[-1].timestamp - timedelta(hours=self.params.window_hours)
        return TimeSeries(data=[p for p in samples.data if p.timestamp > start])

    def predict(self, samples: TimeSeries) -> PredictOutput:
        if not self.weights.fitted:
            raise RuntimeError("Model not fitted")
        if not samples.data:
            raise ValueError("Cannot predict on empty TimeSeries")

        timestamp = samples.data[-1].timestamp
        features = self._featuring(self._scored_span(samples))
        if len(features) < self.params.min_running_samples:
            # Machine (mostly) stopped: no evidence either way.
            return PredictOutput(
                anomaly_status=False, timestamp=timestamp, n_valid=len(features)
            )

        level = np.quantile(features.to_numpy(), self._level_quantile, axis=0)
        z = self._standardize(level[np.newaxis, :])
        scores, contributions = self._detector.score(z)
        score, contributions = float(scores[0]), contributions[0]

        # Scores are quadratic in the deviation; the square root keeps severity
        # proportional to amplitude, so "twice as severe" means what it says.
        severity = float(np.sqrt(max(score, 0.0) / self.weights.threshold))
        # Requirement is *high* vibration: the deviation must be driven by channels
        # above their baseline, not by a machine running quieter than usual.
        points_upward = float(contributions @ np.sign(z[0])) > 0.0

        return PredictOutput(
            anomaly_status=severity >= 1.0
            and (points_upward or not self.params.one_sided),
            timestamp=timestamp,
            severity=severity,
            contributions=self._rank_contributions(contributions),
            n_valid=len(features),
        )

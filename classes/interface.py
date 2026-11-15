from datetime import datetime
from typing import Dict, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat


# --- Domain / data transfer types ---


class DataPoint(BaseModel):
    timestamp: datetime = Field(
        ..., description="Unix timestamp of the time the data point was collected"
    )
    uptime: bool = Field(..., description="Whether the data point is during uptime")
    vel_x: float = Field(
        ..., description="Vibration velocity component along the X axis"
    )
    vel_y: float = Field(
        ..., description="Vibration velocity component along the Y axis"
    )
    vel_z: float = Field(
        ..., description="Vibration velocity component along the Z axis"
    )

    acc_x: float = Field(
        ..., description="Vibration acceleration component along the X axis"
    )
    acc_y: float = Field(
        ..., description="Vibration acceleration component along the Y axis"
    )
    acc_z: float = Field(
        ..., description="Vibration acceleration component along the Z axis"
    )


class TimeSeries(BaseModel):
    data: Sequence[DataPoint] = Field(
        ...,
        description="List of datapoints, ordered in time, of subsequent measurements of some quantity",
    )

    @property
    def length(self) -> int:
        return len(self.data)

    @property
    def last_timestamp(self) -> datetime:
        if not self.data:
            raise ValueError("TimeSeries has no data")
        return self.data[-1].timestamp

    @property
    def first_timestamp(self) -> datetime:
        if not self.data:
            raise ValueError("TimeSeries has no data")
        return self.data[0].timestamp


# --- Model / pipeline types (shared across modules) ---


FEATURE_NAMES: Tuple[str, ...] = ("vel_x", "vel_y", "vel_z", "acc_x", "acc_y", "acc_z")


class Hyperparameters(BaseModel):
    """Base for YAML-backed settings: an unknown key is a typo, so it must fail loudly."""

    model_config = ConfigDict(extra="forbid")


# Default model: sustained level against the asset baseline (classes/anomaly_model.py)


class Weights(BaseModel):
    """Per-asset baseline learned by ``AnomalyModel.fit``."""

    fitted: bool = False
    baseline: Dict[str, PositiveFloat] = Field(
        default_factory=dict,
        description="Upper end of normal operation per feature, in the feature's own unit",
    )
    sampling_period_seconds: PositiveFloat = Field(
        600.0, description="Typical interval between consecutive measurements"
    )


class ModelParams(Hyperparameters):
    baseline_quantile: float = Field(
        0.99,
        gt=0,
        le=1,
        description="Quantile of the running fit samples taken as the upper end of normal",
    )
    horizon_hours: float = Field(
        3.0,
        gt=0,
        description="Trailing part of each window, ending at its last sample, that is scored",
    )
    window_anomaly_ratio: float = Field(
        0.35,
        gt=0,
        le=1,
        description="Share of the horizon that must sit at or above a level for it to count as sustained",
    )
    severity_threshold: float = Field(
        1.5,
        gt=0,
        description="Sustained level, as a multiple of the baseline, from which a window is anomalous",
    )


# Alternative model: PCA with Hotelling T² and SPE (classes/pca_model.py)


class PcaWeights(BaseModel):
    """Per-asset baseline learned by ``PcaAnomalyModel.fit``.

    The detector-specific mapping (the PCA loadings and control limits) lives in
    the detector object; this holds what is common to every detector.
    """

    fitted: bool = False
    center: List[float] = Field(
        default_factory=list,
        description="Robust location (median) of each channel's window level",
    )
    scale: List[float] = Field(
        default_factory=list,
        description="Robust spread (MAD-based sigma) of each channel's window level",
    )
    threshold: float = Field(
        1.0, description="Detector score above which a window is abnormal"
    )


class PcaModelParams(Hyperparameters):
    detector: Literal["pca"] = "pca"

    # --- Window level / baseline robustness ---
    window_hours: float = Field(
        2.0,
        description="Trailing part of each window that is scored, and the window length used to learn the baseline",
    )
    window_anomaly_ratio: float = Field(
        0.2,
        description="Fraction of a window that must be elevated for its level to rise",
    )
    min_running_samples: int = Field(
        4, description="Windows with fewer running samples are not scored"
    )
    min_log_scale: float = Field(
        0.1,
        description="Floor for the robust sigma of a log-level (0.1 ~ 10% amplitude)",
    )
    fit_outlier_z: float = Field(
        6.0,
        description="Fit windows with any |robust z| above this are left out of training",
    )

    # --- Decision rule ---
    threshold_quantile: float = Field(
        0.99, description="Quantile of the fit window scores used as the threshold"
    )
    threshold_margin: float = Field(
        2.0, description="Safety factor applied to that quantile"
    )
    one_sided: bool = Field(
        True, description="Only flag deviations that point to higher vibration"
    )

    # --- PCA detector ---
    pca_variance: float = Field(
        0.9, description="Cumulative variance kept in the principal subspace"
    )


# Default alert engine: time-based events (classes/alert_engine.py)


class AlertParams(Hyperparameters):
    escalation_factor: float = Field(
        2.0,
        gt=1,
        description="Re-alert when severity reaches this multiple of the severity at the last alert",
    )
    rearm_hours: float = Field(
        24.0,
        ge=0,
        description="Time without anomalous windows after which the ongoing event is considered over",
    )


# Alternative alert engine: window-count events (classes/window_alert_engine.py)


class WindowAlertParams(Hyperparameters):
    onset_windows: int = Field(
        3, description="Consecutive abnormal windows required to open an event"
    )
    escalation_ratio: float = Field(
        2.0,
        description="Severity must exceed the level of the last alert by this factor",
    )
    escalation_windows: int = Field(
        2, description="Consecutive windows above the escalated level to re-alert"
    )
    recovery_windows: int = Field(
        24, description="Consecutive normal windows required to close an event"
    )


class PipelineParams(BaseModel):
    model_window_size_hours: float = 4.0
    window_overlap_hours: float = 0.0


class PredictOutput(BaseModel):
    anomaly_status: bool
    timestamp: datetime
    severity: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "How abnormal the window is, in the model's own unit (a multiple of the "
            "baseline for AnomalyModel, of the alarm threshold for PcaAnomalyModel). "
            "None when the model had too little running data to score the window."
        ),
    )
    driver: Optional[str] = Field(
        None, description="Measurement with the largest deviation from its baseline"
    )
    contributions: Dict[str, float] = Field(
        default_factory=dict,
        description="Share of the deviation attributed to each variable, ranked",
    )
    n_valid: int = Field(0, description="Running samples used to score the window")


class AlertDecision(BaseModel):
    alert: bool
    timestamp: datetime
    message: str
    kind: Literal["none", "onset", "escalation"] = Field(
        "none",
        description="Whether the alert opens an event, reports its escalation, or there is no alert",
    )


class TrueIncident(BaseModel):
    start: datetime
    end: datetime

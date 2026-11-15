from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Sequence

import yaml

from .interface import AlertDecision, AlertParams, PredictOutput

DEFAULT_PARAMS_PATH = Path("hyperparameters/alert_hyperparams.yaml")


def load_alert_params(path: Path = DEFAULT_PARAMS_PATH) -> AlertParams:
    """Load alert hyperparameters from YAML. Falls back to AlertParams defaults if file missing."""
    if not path.exists():
        return AlertParams()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return AlertParams(**data)


class AlertEngine:
    """Alerts when an abnormal event starts and again each time it gets markedly worse.

    The engine follows one event at a time. The first anomalous window opens it and
    raises an alert; further anomalous windows stay silent unless severity has grown
    by ``escalation_factor`` since the last alert, which raises a new one. The event
    closes after ``rearm_hours`` without an anomalous window, so a later recurrence is
    reported as a new event.

    Time is read from the prediction timestamps, never from the number of calls, so
    the behaviour does not depend on the window step and repeated windows over a data
    gap do not count as elapsed time.
    """

    def __init__(
        self,
        params: AlertParams | None = None,
        params_path: Path | None = None,
    ):
        self.params = params or load_alert_params(params_path or DEFAULT_PARAMS_PATH)
        self._alerted_severity: float | None = None  # None while no event is open
        self._last_anomaly_at: datetime | None = None

    def predict(
        self, prediction: PredictOutput | Sequence[PredictOutput]
    ) -> AlertDecision:
        """Decide on one prediction, or on a buffered batch of them in time order.

        A batch yields its first alert, or its latest decision when nothing fired.
        """
        if isinstance(prediction, PredictOutput):
            return self._decide(prediction)
        if not prediction:
            raise ValueError("Cannot decide on an empty batch of predictions")

        decisions = [self._decide(p) for p in prediction]
        return next((d for d in decisions if d.alert), decisions[-1])

    def _decide(self, prediction: PredictOutput) -> AlertDecision:
        self._close_event_if_recovered(prediction.timestamp)

        if not prediction.anomaly_status:
            return self._decision(prediction, "No abnormal vibration.")

        self._last_anomaly_at = prediction.timestamp
        # A model that cannot quantify the deviation reports no severity.
        severity = prediction.severity or 0.0
        level = f"{prediction.driver or 'vibration'} at {severity:.1f}x baseline"

        if self._alerted_severity is None:
            self._alerted_severity = severity
            return self._decision(
                prediction, f"Abnormal vibration detected: {level}.", kind="onset"
            )

        if self._has_escalated(severity):
            previous = self._alerted_severity
            self._alerted_severity = severity
            return self._decision(
                prediction,
                f"Abnormal vibration escalated: {level}, up from {previous:.1f}x.",
                kind="escalation",
            )

        return self._decision(prediction, "Ongoing abnormal vibration already alerted.")

    def _close_event_if_recovered(self, now: datetime) -> None:
        if self._alerted_severity is None:
            return
        if now - self._last_anomaly_at >= timedelta(hours=self.params.rearm_hours):
            self._alerted_severity = None

    def _has_escalated(self, severity: float) -> bool:
        # A model that reports no severity gives nothing to compare, so it never escalates.
        return (
            self._alerted_severity > 0
            and severity >= self.params.escalation_factor * self._alerted_severity
        )

    @staticmethod
    def _decision(
        prediction: PredictOutput,
        message: str,
        kind: Literal["none", "onset", "escalation"] = "none",
    ) -> AlertDecision:
        return AlertDecision(
            alert=kind != "none",
            timestamp=prediction.timestamp,
            message=message,
            kind=kind,
        )

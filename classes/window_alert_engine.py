from collections import deque
from pathlib import Path
from typing import Deque, List, Literal

import yaml

from .interface import AlertDecision, PredictOutput, WindowAlertParams

DEFAULT_PARAMS_PATH = Path("hyperparameters/window_alert_hyperparams.yaml")
TOP_CONTRIBUTORS = 3


def load_window_alert_params(path: Path = DEFAULT_PARAMS_PATH) -> WindowAlertParams:
    """Load window-count alert hyperparameters from YAML. Falls back to defaults if file missing."""
    if not path.exists():
        return WindowAlertParams()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return WindowAlertParams(**data)


class WindowCountAlertEngine:
    """Turns the stream of window predictions into alerts, one event at a time.

    Alternative to the default ``AlertEngine``: every rule here counts windows
    rather than hours, which ties its behaviour to the pipeline step.

    An *event* opens after ``onset_windows`` consecutive abnormal windows (one
    alert), stays open while the condition lasts (no further alerts), and closes
    after ``recovery_windows`` consecutive normal windows, which re-arms the engine.

    While an event is open the engine remembers the severity level at the last
    alert. A new alert is raised only when the recent level stays above
    ``escalation_ratio`` times that reference, and the reference then moves up to
    the new level. Comparing levels (rather than the slope of the series) is what
    makes a plateau silent: each rung of the ladder can be climbed only once.
    """

    def __init__(
        self,
        params: WindowAlertParams | None = None,
        params_path: Path | None = None,
    ):
        self.params = params or load_window_alert_params(
            params_path or DEFAULT_PARAMS_PATH
        )
        history = max(self.params.onset_windows, self.params.escalation_windows)
        self._recent: Deque[float] = deque(maxlen=history)
        self._event_open = False
        self._reference_level = 0.0
        self._normal_streak = 0

    @staticmethod
    def _describe(prediction: PredictOutput) -> str:
        top = list(prediction.contributions.items())[:TOP_CONTRIBUTORS]
        return ", ".join(f"{name} {share:.0%}" for name, share in top)

    def _latest(self, n_windows: int) -> List[float] | None:
        """Severity of the last ``n_windows`` consecutive abnormal windows, if that many."""
        if len(self._recent) < n_windows:
            return None
        return list(self._recent)[-n_windows:]

    def _decision(
        self,
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

    def predict(self, prediction: PredictOutput) -> AlertDecision:
        if prediction.severity is None:
            # Machine stopped: no evidence, so the current state is simply held.
            return self._decision(prediction, "Not enough running data; state held.")

        if not prediction.anomaly_status:
            return self._on_normal(prediction)
        return self._on_abnormal(prediction)

    def _on_normal(self, prediction: PredictOutput) -> AlertDecision:
        self._recent.clear()
        if not self._event_open:
            return self._decision(prediction, "No persistent abnormal vibration.")

        self._normal_streak += 1
        if self._normal_streak >= self.params.recovery_windows:
            self._event_open = False
            return self._decision(prediction, "Vibration back to baseline; event closed.")
        return self._decision(prediction, "Event ongoing; waiting for recovery.")

    def _on_abnormal(self, prediction: PredictOutput) -> AlertDecision:
        self._normal_streak = 0
        self._recent.append(prediction.severity)

        if not self._event_open:
            latest = self._latest(self.params.onset_windows)
            if latest is None:
                return self._decision(prediction, "Abnormal window; awaiting confirmation.")
            self._event_open = True
            self._reference_level = sum(latest) / len(latest)
            return self._decision(
                prediction,
                f"Abnormal vibration detected at {self._reference_level:.1f}x the alarm "
                f"threshold ({self._describe(prediction)}).",
                kind="onset",
            )

        level = self._escalated_level()
        if level is None:
            return self._decision(prediction, "Event ongoing; alert already raised.")
        previous, self._reference_level = self._reference_level, level
        return self._decision(
            prediction,
            f"Vibration worsened from {previous:.1f}x to {level:.1f}x the alarm "
            f"threshold ({self._describe(prediction)}).",
            kind="escalation",
        )

    def _escalated_level(self) -> float | None:
        """New reference level if the event got worse since the last alert, else None.

        Worse means the last ``escalation_windows`` windows all sit above
        ``escalation_ratio`` times the level at the last alert.
        """
        latest = self._latest(self.params.escalation_windows)
        if latest is None:
            return None
        if min(latest) < self.params.escalation_ratio * self._reference_level:
            return None
        return sum(latest) / len(latest)

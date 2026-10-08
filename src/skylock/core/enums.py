"""Core enumeration contracts for SkyLock."""

from enum import StrEnum


class TrackState(StrEnum):
    """Tracking state machine states."""

    SEARCH = "SEARCH"
    ACQUIRE = "ACQUIRE"
    TRACK = "TRACK"
    LOST = "LOST"
    REACQUIRE = "REACQUIRE"


class MetricStatus(StrEnum):
    """Metric acquisition and measurement status."""

    NOT_RUN = "NOT_RUN"
    NOT_ACQUIRED = "NOT_ACQUIRED"
    MEASURED = "MEASURED"
    FAILED = "FAILED"


class Verdict(StrEnum):
    """Evaluation verdict against specifications."""

    PASS = "PASS"
    FAIL = "FAIL"
    INDETERMINATE = "INDETERMINATE"


class InputKind(StrEnum):
    """Source input type."""

    SIMULATION = "simulation"
    MP4 = "mp4"
    ORBITAL = "orbital"


class ControlMode(StrEnum):
    """Controller operating mode."""

    AUTO = "AUTO"
    MANUAL = "MANUAL"
    EARTH = "EARTH"
    EARTH_BORESIGHT = "EARTH_BORESIGHT"


class ControlIntentMode(StrEnum):
    """Desired control mode output from tracking pipeline."""

    HOLD = "HOLD"
    TRACK = "TRACK"
    GOTO = "GOTO"

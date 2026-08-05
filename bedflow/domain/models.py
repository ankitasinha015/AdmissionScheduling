from dataclasses import dataclass
from datetime import datetime
from enum import Enum, IntEnum


class CareLevel(IntEnum):
    MEDSURG = 1
    TELEMETRY = 2
    STEPDOWN = 3
    ICU = 4


class BedState(Enum):
    OCCUPIED = "occupied"
    PENDING_DISCHARGE = "pending_discharge"
    DIRTY = "dirty"
    CLEANING = "cleaning"
    AVAILABLE = "available"
    BLOCKED = "blocked"


class Acuity(IntEnum):
    ESI_1 = 1  # sickest. sort ASCENDING for priority.
    ESI_2 = 2
    ESI_3 = 3
    ESI_4 = 4
    ESI_5 = 5


class IsolationType(Enum):
    NONE = "none"
    CONTACT = "contact"
    DROPLET = "droplet"
    AIRBORNE = "airborne"


class Source(Enum):
    PLANNED = "planned"
    ED = "ed"
    TRANSFER = "transfer"


@dataclass(frozen=True)
class Unit:
    id: str
    care_level: CareLevel
    licensed_beds: int
    nurses_on_shift: int
    max_patients_per_nurse: int


@dataclass(frozen=True)
class Bed:
    id: str
    unit_id: str
    care_level: CareLevel
    has_telemetry: bool
    has_negative_pressure: bool
    state: BedState
    expected_free_at: datetime | None


@dataclass(frozen=True)
class AdmissionRequest:
    id: str
    source: Source
    arrived_at: datetime
    scheduled_for: datetime | None
    acuity: Acuity
    required_care_level: CareLevel
    isolation_required: IsolationType
    service: str


@dataclass(frozen=True)
class HospitalState:
    now: datetime
    beds: tuple[Bed, ...]
    units: tuple[Unit, ...]
    pending: tuple[AdmissionRequest, ...]


@dataclass(frozen=True)
class Assignment:
    request_id: str
    bed_id: str
    decided_at: datetime
    wait_minutes: float
    score: float
    rationale: str
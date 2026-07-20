from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import bittensor as bt
import numpy as np

from chronoseek.validator.telemetry import ValidatorTelemetryState


@dataclass
class ValidatorRuntimeState:
    wallet: bt.Wallet
    metagraph: Any
    scores: np.ndarray
    score_lock: Lock
    metagraph_lock: Lock = field(default_factory=Lock)
    miner_endpoints: dict[int, str] = field(default_factory=dict)
    provider_headers: dict[str, str] = field(default_factory=dict)
    responsive_lock: Lock = field(default_factory=Lock)
    responsive_uids: set[int] = field(default_factory=set)
    disqualified_uids: set[int] = field(default_factory=set)
    responsive_initialized: bool = False
    responsive_last_refresh_at: float | None = None
    last_metagraph_sync_block: int | None = None
    telemetry: ValidatorTelemetryState = field(default_factory=ValidatorTelemetryState)

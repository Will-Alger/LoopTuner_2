"""Configuration and environment handling (.env-based secrets, run directories).

Reproducibility: every run records a config + data hash alongside its artifacts so
a model can be traced back to exactly the data and settings that produced it.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


@dataclass
class Settings:
    nightscout_url: str | None
    nightscout_token: str | None
    nightscout_api_secret: str | None
    local_timezone: str
    data_dir: Path
    runs_dir: Path
    anthropic_api_key: str | None

    @classmethod
    def load(cls) -> Settings:
        load_dotenv()
        data_dir = Path(os.environ.get("LOOPTUNER_DATA_DIR", "./data"))
        runs_dir = Path(os.environ.get("LOOPTUNER_RUNS_DIR", "./runs"))
        data_dir.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            nightscout_url=os.environ.get("NIGHTSCOUT_URL") or None,
            nightscout_token=os.environ.get("NIGHTSCOUT_TOKEN") or None,
            nightscout_api_secret=os.environ.get("NIGHTSCOUT_API_SECRET") or None,
            local_timezone=os.environ.get("LOCAL_TIMEZONE", "UTC"),
            data_dir=data_dir,
            runs_dir=runs_dir,
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
        )


def dataframe_hash(frame: pd.DataFrame) -> str:
    """Stable content hash of a dataset frame, for reproducibility provenance."""
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return h.hexdigest()[:16]


def write_run_metadata(path: Path, **meta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        k: (asdict(v) if hasattr(v, "__dataclass_fields__") else v) for k, v in meta.items()
    }
    path.write_text(json.dumps(serializable, indent=2, default=str))

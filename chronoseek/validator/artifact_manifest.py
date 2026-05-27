import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class ArtifactManifestEntry:
    task_id: str
    object_key: str
    public_url: str
    local_path: str
    source_task_hash: str
    encoding_profile: str
    created_at: float
    expires_at: float


class TaskArtifactManifest:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.entries = self._load()

    def _load(self) -> dict[str, ArtifactManifestEntry]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text())
        except Exception:
            return {}

        entries: dict[str, ArtifactManifestEntry] = {}
        for task_id, payload in raw.items():
            try:
                entries[task_id] = ArtifactManifestEntry(**payload)
            except Exception:
                continue
        return entries

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            task_id: asdict(entry)
            for task_id, entry in sorted(self.entries.items())
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    def add(self, entry: ArtifactManifestEntry) -> None:
        self.entries[entry.task_id] = entry
        self.save()

    def active_source_hashes(self, now: float | None = None) -> set[str]:
        current_time = time.time() if now is None else float(now)
        return {
            entry.source_task_hash
            for entry in self.entries.values()
            if entry.expires_at > current_time
        }

    def cleanup_expired(
        self,
        *,
        delete_remote: Callable[[str], None] | None = None,
        now: float | None = None,
    ) -> int:
        current_time = time.time() if now is None else float(now)
        expired = [
            task_id
            for task_id, entry in self.entries.items()
            if entry.expires_at <= current_time
        ]

        for task_id in expired:
            entry = self.entries.pop(task_id)
            local_path = Path(entry.local_path)
            if local_path.exists():
                try:
                    local_path.unlink()
                except OSError:
                    pass
            if delete_remote is not None:
                delete_remote(entry.object_key)

        if expired:
            self.save()
        return len(expired)

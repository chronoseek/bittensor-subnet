from types import SimpleNamespace

import pytest

from chronoseek.validator.artifact_publication import TaskArtifactPublisher
from chronoseek.validator.compression import CompositeCompressor, CompressionResult
from chronoseek.validator.task_models import EncodingProfile
from chronoseek.validator.task_generator_factory import build_validator_task_generator
from chronoseek.validator.task_transforms import TaskTransformPolicy
from chronoseek.video.vidaio import StorageBackedVidaioCompressor


class FakeStorage:
    def __init__(self):
        self.uploads = []

    def upload_file(self, *, local_path: str, object_key: str, content_type: str):
        self.uploads.append((local_path, object_key, content_type))
        return f"https://storage.example/{object_key}"

    def delete_object(self, object_key: str) -> None:
        return None


def test_artifact_publisher_uploads_local_results():
    storage = FakeStorage()
    publisher = TaskArtifactPublisher(storage=storage, artifact_prefix="task-clips/")

    artifact = publisher.publish(
        task_id="task-1",
        compression_result=CompressionResult(
            path="/tmp/task-1.mp4",
            profile_name="profile",
            backend="ffmpeg",
        ),
    )

    assert artifact.object_key == "task-clips/task-1.mp4"
    assert artifact.public_url == "https://storage.example/task-clips/task-1.mp4"
    assert artifact.local_path == "/tmp/task-1.mp4"
    assert storage.uploads == [
        ("/tmp/task-1.mp4", "task-clips/task-1.mp4", "video/mp4")
    ]


def test_artifact_publisher_accepts_hosted_results_without_upload():
    storage = FakeStorage()
    publisher = TaskArtifactPublisher(storage=storage, artifact_prefix="task-clips")

    artifact = publisher.publish(
        task_id="task-1",
        compression_result=CompressionResult(
            path="",
            profile_name="profile",
            backend="vidaio",
            public_url="https://vidaio.example/task-1.mp4",
        ),
    )

    assert artifact.public_url == "https://vidaio.example/task-1.mp4"
    assert artifact.object_key == ""
    assert artifact.local_path == ""
    assert storage.uploads == []


def test_artifact_publisher_rejects_empty_compression_result():
    publisher = TaskArtifactPublisher(
        storage=FakeStorage(),
        artifact_prefix="task-clips",
    )

    with pytest.raises(RuntimeError, match="local path or public URL"):
        publisher.publish(
            task_id="task-1",
            compression_result=CompressionResult(
                path="",
                profile_name="profile",
                backend="invalid",
            ),
        )


def test_transform_policy_always_builds_variants_from_configured_base():
    base_profile = EncodingProfile(name="base", video_bitrate="1000k")
    policy = TaskTransformPolicy(
        base_profile=base_profile,
        enable_profile_variants=True,
    )
    target = SimpleNamespace(encoding_profile=base_profile)

    class CompactRng:
        @staticmethod
        def choice(options):
            return options[1]

    first = policy.select_profile(rng=CompactRng())
    policy.apply_profile(first, target)
    second = policy.select_profile(rng=CompactRng())

    assert first == second
    assert second.name == "base:compact"
    assert second.video_bitrate == "800k"
    assert target.encoding_profile == second


def test_compressor_wrappers_propagate_encoding_profile_to_children():
    original = EncodingProfile(name="original")
    selected = EncodingProfile(name="selected")
    local = SimpleNamespace(encoding_profile=original)
    vidaio = SimpleNamespace(encoding_profile=original)
    storage_backed = StorageBackedVidaioCompressor(
        vidaio_compressor=vidaio,
        storage=FakeStorage(),
    )
    composite = CompositeCompressor(
        local_compressor=local,
        preferred_compressor=storage_backed,
        preferred_enabled=True,
    )

    composite.encoding_profile = selected

    assert composite.encoding_profile == selected
    assert local.encoding_profile == selected
    assert storage_backed.encoding_profile == selected
    assert vidaio.encoding_profile == selected


def _source_config(**overrides):
    values = {
        "task_dataset_path": "",
        "task_split": "validation",
        "hf_cache_dir": "",
        "hf_activitynet_filename": "",
        "require_accessible_videos": False,
        "task_max_sampling_attempts": 1,
        "enable_hardened_tasks": False,
        "validator_task_secret": "",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_task_generator_factory_returns_legacy_generator_when_hardening_is_off(
    monkeypatch,
):
    legacy_generator = SimpleNamespace(dataset=[])
    monkeypatch.setattr(
        "chronoseek.validator.task_generator_factory.task_gen_module.ActivityNetTaskGenerator",
        lambda **kwargs: legacy_generator,
    )

    result = build_validator_task_generator(
        config=_source_config(),
        availability_checker=None,
        validator_hotkey="hotkey",
        current_block_provider=lambda: 1,
    )

    assert result is legacy_generator


def test_task_generator_factory_requires_secret_when_hardening_is_on(monkeypatch):
    monkeypatch.setattr(
        "chronoseek.validator.task_generator_factory.task_gen_module.ActivityNetTaskGenerator",
        lambda **kwargs: SimpleNamespace(dataset=[]),
    )

    with pytest.raises(ValueError, match="VALIDATOR_TASK_SECRET"):
        build_validator_task_generator(
            config=_source_config(enable_hardened_tasks=True),
            availability_checker=None,
            validator_hotkey="hotkey",
            current_block_provider=lambda: 1,
        )

import asyncio
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import httpx

from chronoseek.chutes.runtime import ChutesRuntimeEndpoint
from chronoseek.protocol_models import VideoSearchResult
from chronoseek.protocol_models import VideoSearchResponse
from chronoseek.scoring import score_response
from chronoseek.validator.forward import MinerQueryResult, run_step
from chronoseek.validator.artifact_manifest import (
    ArtifactManifestEntry,
    TaskArtifactManifest,
)
from chronoseek.validator.clipper import ClipResult
from chronoseek.validator.compression import CompressionResult
from chronoseek.validator.hardened_task_gen import (
    HardenedActivityNetTaskGenerator,
    HardenedTaskGeneratorConfig,
)
from chronoseek.validator.query_variants import QueryVariantSelector
from chronoseek.validator.task_models import CropPlan, EncodingProfile, ValidationTask
from chronoseek.validator.task_models import normalize_generated_task
from chronoseek.validator.task_sampler import ActivityNetTaskSampler


def test_normalize_generated_task_supports_legacy_tuple_and_validation_task():
    legacy = normalize_generated_task(
        ("https://example.com/video.mp4", "find a dog", [(1, 2)]),
        default_top_k=1,
    )
    assert legacy.video_url == "https://example.com/video.mp4"
    assert legacy.query == "find a dog"
    assert legacy.ground_truths == [(1.0, 2.0)]
    assert legacy.top_k == 1

    task = ValidationTask(
        task_id="task-1",
        request_id="request-1",
        video_url="https://hippius.example/task.mp4",
        query="query variant",
        ground_truths=[(0.5, 1.5)],
        clip_duration=10.0,
        source_video_id="video-1",
        source_caption_id="caption-1",
        crop_start=10.0,
        crop_end=20.0,
        query_variant_id="private:caption-1:0",
        artifact_url="https://hippius.example/task.mp4",
        artifact_key="validator-tasks/task.mp4",
        expires_at=time.time() + 3600,
    )
    normalized = normalize_generated_task(task, default_top_k=1)
    assert normalized.video_url == task.video_url
    assert normalized.request_id == "request-1"
    assert normalized.clip_duration == 10.0
    assert normalized.query_variant_id == "private:caption-1:0"


def test_score_response_scores_first_result_only_and_rejects_oversized_clip_answer():
    predictions = [
        VideoSearchResult(start=0.0, end=80.0, confidence=1.0),
        VideoSearchResult(start=5.0, end=10.0, confidence=0.1),
    ]
    assert (
        score_response(
            predictions,
            [(5.0, 10.0)],
            latency=0.1,
            clip_duration=90.0,
            max_prediction_duration_seconds=60.0,
            score_top_k=1,
        )
        == 0.0
    )

    assert (
        score_response(
            predictions[1:],
            [(5.0, 10.0)],
            latency=0.1,
            clip_duration=90.0,
            max_prediction_duration_seconds=60.0,
            score_top_k=1,
        )
        == 1.0
    )


def test_query_variants_prefers_private_manifest_and_falls_back(tmp_path):
    variants_path = tmp_path / "variants.json"
    variants_path.write_text(
        json.dumps({"caption-id": ["private wording A", "private wording B"]}),
        encoding="utf-8",
    )
    selector = QueryVariantSelector(str(variants_path))
    sampler = ActivityNetTaskSampler(
        [
            {
                "task_id": "video-1",
                "video_url": "https://example.com/source.mp4",
                "caption_intervals": {"raw caption": [(1, 2)]},
            }
        ],
        validator_hotkey="hotkey",
        task_secret="secret",
        video_cooldown_seconds=0,
        caption_cooldown_seconds=0,
    )
    rng = sampler.make_rng(block=1, nonce=1)
    private_variant = selector.select(
        caption="raw caption",
        source_caption_id="caption-id",
        rng=rng,
    )
    assert private_variant.source == "private"
    assert private_variant.query.startswith("private wording")

    fallback_variant = selector.select(
        caption="raw caption",
        source_caption_id="missing-caption-id",
        rng=rng,
    )
    assert fallback_variant.source == "fallback"
    assert fallback_variant.query != "raw caption"


def test_task_sampler_is_secret_seeded_and_respects_caption_cooldown():
    dataset = [
        {
            "task_id": "video-1",
            "video_url": "https://example.com/source.mp4",
            "caption_intervals": {"raw caption": [(1, 2)]},
        }
    ]
    sampler_a = ActivityNetTaskSampler(
        dataset,
        validator_hotkey="hotkey",
        task_secret="secret",
        video_cooldown_seconds=0,
        caption_cooldown_seconds=0,
    )
    sampler_b = ActivityNetTaskSampler(
        dataset,
        validator_hotkey="hotkey",
        task_secret="secret",
        video_cooldown_seconds=0,
        caption_cooldown_seconds=0,
    )

    sample_a, _ = sampler_a.sample(block=100, nonce=1)
    sample_b, _ = sampler_b.sample(block=100, nonce=1)
    assert sample_a == sample_b

    cooldown_sampler = ActivityNetTaskSampler(
        dataset,
        validator_hotkey="hotkey",
        task_secret="secret",
        video_cooldown_seconds=0,
        caption_cooldown_seconds=3600,
    )
    cooldown_sampler.sample(block=100, nonce=1)
    try:
        cooldown_sampler.sample(block=100, nonce=2)
    except RuntimeError as exc:
        assert "cooldown" in str(exc)
    else:
        raise AssertionError("caption cooldown should block immediate replay")


def test_artifact_manifest_records_active_hashes_and_cleans_expired(tmp_path):
    local_file = tmp_path / "artifact.mp4"
    local_file.write_bytes(b"video")
    manifest = TaskArtifactManifest(tmp_path / "manifest.json")
    deleted_remote = []
    now = time.time()
    manifest.add(
        ArtifactManifestEntry(
            task_id="expired-task",
            object_key="validator-tasks/expired-task.mp4",
            public_url="https://public.example/expired-task.mp4",
            local_path=str(local_file),
            source_task_hash="source-hash",
            encoding_profile="h264-720p-v1",
            created_at=now - 100,
            expires_at=now - 1,
        )
    )

    assert manifest.active_source_hashes(now=now) == set()
    assert (
        manifest.cleanup_expired(
            delete_remote=lambda key: deleted_remote.append(key),
            now=now,
        )
        == 1
    )
    assert not local_file.exists()
    assert deleted_remote == ["validator-tasks/expired-task.mp4"]


class FakeClipper:
    def __init__(self, root: Path):
        self.root = root
        self.encoding_profile = EncodingProfile()

    def create_clip(self, **kwargs):
        task_id = kwargs["task_id"]
        path = self.root / f"{task_id}.mp4"
        path.write_bytes(b"clip")
        return ClipResult(
            path=str(path),
            crop_plan=CropPlan(
                source_start=10.0,
                source_end=25.0,
                clip_duration=15.0,
                shifted_ground_truths=[(2.0, 5.0)],
            ),
            crop_bucket="10-20",
        )


class FakeCompressor:
    def compress(self, *, input_path: str, output_path: str):
        shutil.copyfile(input_path, output_path)
        return CompressionResult(
            path=output_path,
            profile_name="h264-720p-v1",
            backend="ffmpeg",
        )


class FakeStorage:
    def __init__(self):
        self.uploads = []
        self.deleted = []

    def upload_file(self, *, local_path: str, object_key: str, content_type: str):
        self.uploads.append((local_path, object_key, content_type))
        return f"https://hippius.example/{object_key}"

    def delete_object(self, object_key: str):
        self.deleted.append(object_key)


def test_hardened_generator_outputs_clip_local_protocol_task(tmp_path):
    variants_path = tmp_path / "variants.json"
    variants_path.write_text(json.dumps({"raw caption": ["private query"]}))
    sampler = ActivityNetTaskSampler(
        [
            {
                "task_id": "video-1",
                "video_url": "https://example.com/source.mp4",
                "caption_intervals": {"raw caption": [(12, 15)]},
            }
        ],
        validator_hotkey="hotkey",
        task_secret="secret",
        video_cooldown_seconds=0,
        caption_cooldown_seconds=0,
    )
    manifest = TaskArtifactManifest(tmp_path / "manifest.json")
    storage = FakeStorage()
    generator = HardenedActivityNetTaskGenerator(
        sampler=sampler,
        query_selector=QueryVariantSelector(str(variants_path)),
        clipper=FakeClipper(tmp_path),
        compressor=FakeCompressor(),
        storage=storage,
        manifest=manifest,
        validator_hotkey="hotkey",
        current_block_provider=lambda: 123,
        config=HardenedTaskGeneratorConfig(
            artifact_prefix="validator-tasks",
            ttl_hours=6,
            cleanup_interval_seconds=900,
            max_generation_attempts=1,
        ),
    )

    task = generator.generate_task()

    assert isinstance(task, ValidationTask)
    assert task.video_url.startswith("https://hippius.example/validator-tasks/")
    assert task.query == "private query"
    assert task.ground_truths == [(2.0, 5.0)]
    assert task.clip_duration == 15.0
    assert task.source_video_id == "video-1"
    assert storage.uploads[0][1] == task.artifact_key
    assert task.task_id in manifest.entries


def test_run_step_accepts_validation_task_and_sends_top_k_one(monkeypatch):
    task = ValidationTask(
        task_id="task-1",
        request_id="request-1",
        video_url="https://hippius.example/task.mp4",
        query="private query",
        ground_truths=[(2.0, 5.0)],
        clip_duration=15.0,
        source_video_id="video-1",
        source_caption_id="caption-1",
        crop_start=10.0,
        crop_end=25.0,
        query_variant_id="private:caption-1:0",
        artifact_url="https://hippius.example/task.mp4",
        artifact_key="validator-tasks/task.mp4",
        expires_at=time.time() + 3600,
    )

    class StaticTaskGenerator:
        def generate_task(self):
            return task

    captured = {}

    async def fake_query_miner(*args, **kwargs):
        request_model = args[4]
        captured["request"] = request_model
        return MinerQueryResult(
            response=VideoSearchResponse(
                request_id=request_model.request_id,
                results=[VideoSearchResult(start=2.0, end=5.0, confidence=0.9)],
            ),
            latency=0.1,
        )

    monkeypatch.setattr("chronoseek.validator.forward.query_miner", fake_query_miner)

    async def execute():
        async with httpx.AsyncClient() as client:
            return await run_step(
                StaticTaskGenerator(),
                SimpleNamespace(uids=[0], hotkeys=["hotkey-1"]),
                wallet=SimpleNamespace(),
                client=client,
                miner_endpoints=[
                    ChutesRuntimeEndpoint(
                        uid=0,
                        hotkey="hotkey-1",
                        endpoint="https://runtime.example",
                    )
                ],
                validator_eval_top_k=1,
                max_prediction_duration_seconds=60.0,
            )

    scores = asyncio.run(execute())

    assert scores == [(0, 1.0)]
    assert captured["request"].video_url == "https://hippius.example/task.mp4"
    assert captured["request"].query == "private query"
    assert captured["request"].top_k == 1
    assert captured["request"].request_id == "request-1"

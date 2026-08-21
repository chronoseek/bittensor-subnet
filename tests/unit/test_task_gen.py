import json
from pathlib import Path
from unittest.mock import patch

from chronoseek.constants import DEFAULT_TASK_DATASET_HIPPIUS_REPO_ID
from chronoseek.validator.task_gen import ActivityNetTaskGenerator
from chronoseek.validator.video_availability import VideoAvailabilityResult


def test_local_manifest_loads_validation_split(tmp_path):
    dataset_path = tmp_path / "activitynet.json"
    dataset_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "task-1",
                        "split": "validation",
                        "difficulty": "easy",
                        "video_url": "https://example.com/video.mp4",
                        "query": "a person opens the door",
                        "ground_truths": [
                            {"start": 1.0, "end": 2.5},
                            {"start": 4.0, "end": 5.0},
                        ],
                    }
                ]
            }
        )
    )

    task_gen = ActivityNetTaskGenerator(dataset_path=str(dataset_path))

    assert task_gen.dataset
    assert all(task["split"] == "validation" for task in task_gen.dataset)
    assert all("task_id" in task for task in task_gen.dataset)


def test_cache_dir_expands_user_before_huggingface_download(monkeypatch, tmp_path):
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    dataset_path = snapshot_dir / "validation.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "video_id": "v_demo1234567",
                    "caption": "a person waves",
                    "start_time": 1.0,
                    "end_time": 2.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HF_TOKEN", "hf_test")

    with patch(
        "hippius_hub.snapshot_download",
        side_effect=RuntimeError("hippius dataset unavailable in test"),
    ), patch(
        "huggingface_hub.snapshot_download",
        return_value=str(snapshot_dir),
    ) as snapshot_download:
        task_gen = ActivityNetTaskGenerator(cache_dir="~/.cache/huggingface")

    assert task_gen.cache_dir == str(Path("~/.cache/huggingface").expanduser())
    assert snapshot_download.call_args.kwargs["cache_dir"] == str(
        Path("~/.cache/huggingface").expanduser()
    )
    assert not snapshot_download.call_args.kwargs["cache_dir"].startswith("~")


def test_dataset_loads_from_hippius_before_huggingface(monkeypatch, tmp_path):
    snapshot_dir = tmp_path / "hippius-snapshot"
    snapshot_dir.mkdir()
    dataset_path = snapshot_dir / "validation.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "video_id": "v_demo1234567",
                    "caption": "a person waves",
                    "start_time": 1.0,
                    "end_time": 2.0,
                }
            ]
        ),
        encoding="utf-8",
    )

    with patch(
        "hippius_hub.snapshot_download",
        return_value=str(snapshot_dir),
    ) as hippius_snapshot_download, patch(
        "huggingface_hub.snapshot_download",
    ) as hf_snapshot_download:
        task_gen = ActivityNetTaskGenerator()

    hippius_snapshot_download.assert_called_once()
    assert (
        hippius_snapshot_download.call_args.kwargs["repo_id"]
        == DEFAULT_TASK_DATASET_HIPPIUS_REPO_ID
    )
    hf_snapshot_download.assert_not_called()
    assert task_gen.dataset


def test_dataset_falls_back_to_bundled_local_dataset_when_all_remote_sources_fail(
    monkeypatch,
):
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with patch(
        "hippius_hub.snapshot_download",
        side_effect=RuntimeError("hippius dataset unavailable in test"),
    ), patch(
        "huggingface_hub.snapshot_download",
        side_effect=RuntimeError("huggingface dataset unavailable in test"),
    ):
        task_gen = ActivityNetTaskGenerator()

    assert task_gen.dataset


def test_huggingface_dataset_loads_without_token(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    snapshot_dir = tmp_path / "hf-snapshot"
    snapshot_dir.mkdir()
    dataset_path = snapshot_dir / "validation.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "video_id": "v_demo1234567",
                    "caption": "a person waves",
                    "start_time": 1.0,
                    "end_time": 2.0,
                }
            ]
        ),
        encoding="utf-8",
    )

    with patch(
        "hippius_hub.snapshot_download",
        side_effect=RuntimeError("hippius dataset unavailable in test"),
    ), patch(
        "huggingface_hub.snapshot_download",
        return_value=str(snapshot_dir),
    ) as hf_snapshot_download:
        task_gen = ActivityNetTaskGenerator()

    assert hf_snapshot_download.call_args.kwargs["token"] is None
    assert task_gen.dataset


def test_generate_task_returns_expected_shape(tmp_path):
    dataset_path = tmp_path / "activitynet.json"
    dataset_path.write_text(
        json.dumps(
            {
                "database": {
                    "video-1": {
                        "url": "https://example.com/video.mp4",
                        "sentences": [
                            "a person opens the door",
                            "a person walks into the room",
                        ],
                        "timestamps": [[1.0, 2.5], [3.0, 4.5]],
                    }
                }
            }
        )
    )

    task_gen = ActivityNetTaskGenerator(dataset_path=str(dataset_path))

    video_url, query, ground_truths = task_gen.generate_task()

    assert video_url.startswith("https://")
    assert isinstance(query, str) and query
    assert isinstance(ground_truths, list)
    assert ground_truths
    assert all(
        isinstance(interval, tuple) and len(interval) == 2 for interval in ground_truths
    )


def test_generate_task_keeps_all_matching_intervals_for_caption(tmp_path):
    dataset_path = tmp_path / "activitynet.json"
    dataset_path.write_text(
        json.dumps(
            {
                "database": {
                    "video-1": {
                        "url": "https://example.com/video.mp4",
                        "sentences": [
                            "a person opens the door",
                            "a person opens the door",
                            "a person walks into the room",
                        ],
                        "timestamps": [[1.0, 2.5], [4.0, 5.5], [7.0, 8.5]],
                    }
                }
            }
        )
    )

    task_gen = ActivityNetTaskGenerator(dataset_path=str(dataset_path))
    video_url, query, ground_truths = task_gen.generate_task()

    assert video_url == "https://example.com/video.mp4"
    if query == "a person opens the door":
        assert ground_truths == [(1.0, 2.5), (4.0, 5.5)]
    else:
        assert query == "a person walks into the room"
        assert ground_truths == [(7.0, 8.5)]


def test_row_oriented_dataset_loads_single_and_multiple_interval_formats(tmp_path):
    dataset_path = tmp_path / "activitynet_rows.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "video_id": "v_demo1234567",
                    "caption": "a person waves",
                    "start_time": 1.0,
                    "end_time": 2.0,
                },
                {
                    "video_id": "v_demo1234567",
                    "caption": "a person waves",
                    "ground_truth": [3.0, 4.0],
                },
                {
                    "video_id": "v_demo1234567",
                    "caption": "a person waves",
                    "ground_truths": [
                        {"start": 5.0, "end": 6.0},
                        [7.0, 8.0],
                    ],
                },
            ]
        )
    )

    task_gen = ActivityNetTaskGenerator(dataset_path=str(dataset_path))
    assert len(task_gen.dataset) == 1

    video_url, query, ground_truths = task_gen.generate_task()
    assert video_url == "https://www.youtube.com/watch?v=demo1234567"
    assert query == "a person waves"
    assert ground_truths == [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]


def test_row_with_sentences_and_timestamps_splits_into_distinct_captions(tmp_path):
    """Regression test: the real ActivityNet_Captions HF mirror stores each
    video as one row with a combined `caption` (all sentences concatenated)
    plus parallel `sentences`/`timestamps` arrays, one timestamp per
    sentence. The row must be split per-sentence - pairing the combined
    caption with the whole `timestamps` array instead collapses a video's
    several distinct captions/events into a single fake caption entry with
    unrelated intervals attached, and defeats per-caption sampling/shuffling
    (only one caption ever exists per video)."""
    dataset_path = tmp_path / "activitynet_rows.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "video_id": "v_1cU8sp05Bu0",
                    "caption": (
                        "street is shown with different cas passing by the "
                        "street. man is tanding in a music room playing "
                        "congas. people are walking in the sidewalk by a "
                        "store."
                    ),
                    "timestamps": [[0, 11.34], [11.34, 64.81], [0, 11.02]],
                    "sentences": [
                        "street is shown with different cas passing by the street.",
                        "man is tanding in a music room playing congas.",
                        "people are walking in the sidewalk by a store.",
                    ],
                },
            ]
        )
    )

    task_gen = ActivityNetTaskGenerator(dataset_path=str(dataset_path))
    assert len(task_gen.dataset) == 1

    caption_intervals = task_gen.dataset[0]["caption_intervals"]
    assert caption_intervals == {
        "street is shown with different cas passing by the street.": [
            (0.0, 11.34)
        ],
        "man is tanding in a music room playing congas.": [(11.34, 64.81)],
        "people are walking in the sidewalk by a store.": [(0.0, 11.02)],
    }


def test_resolve_snapshot_dataset_file_accepts_row_json(tmp_path):
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    dataset_path = snapshot_dir / "validation.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "video_id": "v_demo1234567",
                    "caption": "a person waves",
                    "start_time": 1.0,
                    "end_time": 2.0,
                }
            ]
        )
    )

    task_gen = ActivityNetTaskGenerator(dataset_path=str(dataset_path))
    resolved = task_gen._resolve_snapshot_dataset_file(str(snapshot_dir))

    assert resolved == str(dataset_path)


class StubAvailabilityChecker:
    def __init__(self, statuses, accessible_urls=None):
        self.statuses = statuses
        self.refreshed = 0
        self.accessible_urls = list(accessible_urls or [])

    def check(self, url):
        return self.statuses[url]

    def refresh_unavailable(self):
        self.refreshed += 1
        return len(
            [status for status in self.statuses.values() if not status.accessible]
        )

    def get_accessible_urls(self):
        return list(self.accessible_urls)


def test_generate_task_skips_inaccessible_videos(tmp_path):
    dataset_path = tmp_path / "activitynet.json"
    dataset_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "bad-video",
                        "split": "validation",
                        "difficulty": "easy",
                        "video_url": "https://example.com/bad.mp4",
                        "query": "bad query",
                        "ground_truth": {"start": 1.0, "end": 2.0},
                    },
                    {
                        "task_id": "good-video",
                        "split": "validation",
                        "difficulty": "easy",
                        "video_url": "https://example.com/good.mp4",
                        "query": "good query",
                        "ground_truth": {"start": 3.0, "end": 4.0},
                    },
                ]
            }
        )
    )

    checker = StubAvailabilityChecker(
        {
            "https://example.com/bad.mp4": VideoAvailabilityResult(
                accessible=False, reason="private"
            ),
            "https://example.com/good.mp4": VideoAvailabilityResult(
                accessible=True, reason="ok"
            ),
        }
    )

    task_gen = ActivityNetTaskGenerator(
        dataset_path=str(dataset_path),
        require_accessible_videos=True,
        availability_checker=checker,
        max_sampling_attempts=2,
    )

    video_url, query, ground_truths = task_gen.generate_task()
    assert video_url == "https://example.com/good.mp4"
    assert query == "good query"
    assert ground_truths == [(3.0, 4.0)]


def test_refresh_video_lookup_delegates_to_availability_checker(tmp_path):
    dataset_path = tmp_path / "activitynet.json"
    dataset_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "bad-video",
                        "split": "validation",
                        "difficulty": "easy",
                        "video_url": "https://example.com/bad.mp4",
                        "query": "bad query",
                        "ground_truth": {"start": 1.0, "end": 2.0},
                    }
                ]
            }
        )
    )

    checker = StubAvailabilityChecker(
        {
            "https://example.com/bad.mp4": VideoAvailabilityResult(
                accessible=False, reason="private"
            ),
        }
    )

    task_gen = ActivityNetTaskGenerator(
        dataset_path=str(dataset_path),
        require_accessible_videos=True,
        availability_checker=checker,
    )

    removed_entries = task_gen.refresh_video_lookup()
    assert removed_entries == 1
    assert checker.refreshed == 1


def test_generate_task_falls_back_to_cached_accessible_video(tmp_path, monkeypatch):
    dataset_path = tmp_path / "activitynet.json"
    dataset_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "bad-video",
                        "split": "validation",
                        "difficulty": "easy",
                        "video_url": "https://example.com/bad.mp4",
                        "query": "bad query",
                        "ground_truth": {"start": 1.0, "end": 2.0},
                    },
                    {
                        "task_id": "good-video",
                        "split": "validation",
                        "difficulty": "easy",
                        "video_url": "https://example.com/good.mp4",
                        "query": "good query",
                        "ground_truth": {"start": 3.0, "end": 4.0},
                    },
                ]
            }
        )
    )

    checker = StubAvailabilityChecker(
        {
            "https://example.com/bad.mp4": VideoAvailabilityResult(
                accessible=False, reason="private"
            ),
        },
        accessible_urls=["https://example.com/good.mp4"],
    )

    task_gen = ActivityNetTaskGenerator(
        dataset_path=str(dataset_path),
        require_accessible_videos=True,
        availability_checker=checker,
        max_sampling_attempts=1,
    )

    monkeypatch.setattr("random.sample", lambda population, k: [population[0]])

    video_url, query, ground_truths = task_gen.generate_task()

    assert video_url == "https://example.com/good.mp4"
    assert query == "good query"
    assert ground_truths == [(3.0, 4.0)]

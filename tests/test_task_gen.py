import json

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

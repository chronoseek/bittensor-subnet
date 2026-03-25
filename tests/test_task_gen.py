import json

from chronoseek.validator.task_gen import ActivityNetTaskGenerator


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
    assert all(isinstance(interval, tuple) and len(interval) == 2 for interval in ground_truths)


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

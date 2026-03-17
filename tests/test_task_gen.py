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
                        "ground_truth": {"start": 1.0, "end": 2.5},
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

    video_url, query, ground_truth = task_gen.generate_task()

    assert video_url.startswith("https://")
    assert isinstance(query, str) and query
    assert isinstance(ground_truth, tuple)
    assert len(ground_truth) == 2

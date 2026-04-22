import numpy as np

from chronoseek.miner.logic import MinerLogic
from chronoseek.miner.utils.transcript_engine import TranscriptSegment


class StubMlEngine:
    def __init__(self, scores):
        self.scores = np.array(scores, dtype=np.float32)

    def compute_text_similarity(self, query, texts):
        return np.array(self.scores[: len(texts)], dtype=np.float32)


def make_logic():
    logic = MinerLogic.__new__(MinerLogic)
    logic.ml_engine = StubMlEngine([0.25, 0.8])
    return logic


def test_score_transcript_segments_uses_text_similarity_scores():
    logic = make_logic()
    segments = [
        TranscriptSegment(start=0.0, end=2.0, text="hello there"),
        TranscriptSegment(start=2.0, end=5.0, text="general kenobi"),
    ]

    scored = logic._score_transcript_segments("hello", segments)

    assert [(start, end, text) for start, end, _, text in scored] == [
        (0.0, 2.0, "hello there"),
        (2.0, 5.0, "general kenobi"),
    ]
    np.testing.assert_allclose(
        np.array([score for _, _, score, _ in scored], dtype=np.float32),
        np.array([0.25, 0.8], dtype=np.float32),
    )


def test_fuse_visual_audio_timeline_returns_vision_only_without_audio():
    logic = make_logic()
    visual_ts = (0.0, 1.0, 2.0)
    visual_scores = np.array([0.2, 0.4, 0.6], dtype=np.float32)

    fused_ts, fused_scores, fusion_mode = logic._fuse_visual_audio_timeline(
        visual_ts,
        visual_scores,
        [],
    )

    assert fused_ts == visual_ts
    np.testing.assert_allclose(fused_scores, visual_scores)
    assert fusion_mode == "vision_only"


def test_fuse_visual_audio_timeline_applies_audio_as_boost_only():
    logic = make_logic()
    visual_ts = (0.0, 1.0, 2.0)
    visual_scores = np.array([0.2, 0.4, 0.6], dtype=np.float32)
    audio_segments = [
        (0.0, 1.0, 1.0, "hello there"),
    ]

    fused_ts, fused_scores, fusion_mode = logic._fuse_visual_audio_timeline(
        visual_ts,
        visual_scores,
        audio_segments,
    )

    assert fused_ts == (0.0, 1.0, 2.0)
    np.testing.assert_allclose(
        fused_scores,
        np.array(
            [
                0.2 + 0.25 * (1.0 - 0.2),
                0.4 + 0.25 * (1.0 - 0.4),
                0.6,
            ],
            dtype=np.float32,
        ),
    )
    assert fusion_mode == "vision_audio"

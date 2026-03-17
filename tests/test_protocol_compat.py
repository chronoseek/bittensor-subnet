import json
from pathlib import Path

import jsonschema

from chronoseek.protocol_models import (
    ProtocolError,
    VideoSearchRequest,
    VideoSearchResponse,
)

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_DIR = ROOT / "protocol_artifacts"


def load_schema(name: str) -> dict:
    return json.loads((SCHEMA_DIR / name).read_text())


def test_video_search_request_matches_protocol_schema():
    payload = VideoSearchRequest(
        video={"url": "https://example.com/video.mp4"},
        query="the moment the speaker writes on the whiteboard",
    ).model_dump(mode="json", exclude_none=True)

    jsonschema.validate(payload, load_schema("video-search-request.schema.json"))


def test_video_search_response_matches_protocol_schema():
    payload = VideoSearchResponse(
        status="completed",
        results=[
            {
                "start": 12.5,
                "end": 20.0,
                "confidence": 0.91,
            }
        ],
        miner_metadata={"source": "validator-gateway"},
    ).model_dump(mode="json", exclude_none=True)

    jsonschema.validate(payload, load_schema("video-search-response.schema.json"))


def test_protocol_error_matches_protocol_schema():
    payload = ProtocolError(
        error={
            "code": "VIDEO_FETCH_FAILED",
            "message": "The video URL could not be fetched.",
        }
    ).model_dump(mode="json", exclude_none=True)

    jsonschema.validate(payload, load_schema("protocol-error.schema.json"))

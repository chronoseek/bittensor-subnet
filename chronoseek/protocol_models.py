from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class VideoSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: HttpUrl


class VideoSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_version: Literal["2026-03-01"] = "2026-03-01"
    request_id: str | None = None
    video: VideoSource
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_video_url(cls, data: Any):
        if isinstance(data, dict) and "video_url" in data and "video" not in data:
            return {
                **data,
                "video": {
                    "url": data["video_url"],
                },
            }

        return data

    @property
    def video_url(self) -> str:
        return str(self.video.url)


class VideoSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: float = Field(ge=0)
    end: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_interval(self):
        if self.end < self.start:
            raise ValueError("end must be greater than or equal to start")
        return self


class VideoSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_version: Literal["2026-03-01"] = "2026-03-01"
    request_id: str | None = None
    status: Literal["accepted", "processing", "completed", "failed"] = "completed"
    results: list[VideoSearchResult] = Field(default_factory=list)
    miner_metadata: dict[str, Any] | None = None


class ProtocolErrorPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: Literal[
        "INVALID_REQUEST",
        "UNSUPPORTED_PROTOCOL_VERSION",
        "VIDEO_FETCH_FAILED",
        "VIDEO_UNREADABLE",
        "QUERY_INVALID",
        "INFERENCE_FAILED",
        "TIMEOUT",
        "INTERNAL_ERROR",
    ]
    message: str = Field(min_length=1)
    details: dict[str, Any] | None = None


class ProtocolError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_version: Literal["2026-03-01"] = "2026-03-01"
    error: ProtocolErrorPayload

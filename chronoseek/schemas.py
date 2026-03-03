from pydantic import BaseModel, Field
from typing import List, Optional
import bittensor as bt


class VideoSearchResult(BaseModel):
    start: float = Field(..., description="Start timestamp in seconds")
    end: float = Field(..., description="End timestamp in seconds")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


# class VideoSearchRequest(bt.Synapse):
class VideoSearchRequest(BaseModel):
    video_url: str = Field(..., description="URL of the video to analyze")
    query: str = Field(..., description="Natural language query describing the moment")


# class VideoSearchResponse(bt.Synapse):
class VideoSearchResponse(BaseModel):
    results: List[VideoSearchResult] = Field(default_factory=list)
    miner_metadata: Optional[dict] = Field(
        default=None, description="Optional debug info"
    )

import time

import requests

from chronoseek.validator.compression import CompressionUnavailable


class VidaioApiClient:
    """Small client for Vidaio's asynchronous public compression workflow."""

    TERMINAL_FAILURES = {"failed", "cancelled", "canceled", "trigger_failed"}

    def __init__(
        self,
        *,
        api_base_url: str,
        api_key: str | None,
        timeout_seconds: float,
        compression_type: str = "High",
        tier: str = "Standard",
        poll_interval_seconds: float = 15.0,
    ):
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.compression_type = compression_type
        self.tier = tier
        self.poll_interval_seconds = max(0.25, float(poll_interval_seconds))

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key} if self.api_key else {}

    @staticmethod
    def _format_payload(payload: dict) -> str:
        keys = ("task_id", "status", "message", "error", "detail", "reason")
        parts = [
            f"{key}={payload[key]}"
            for key in keys
            if payload.get(key) is not None and payload.get(key) != ""
        ]
        return "; ".join(parts) if parts else repr(payload)

    @classmethod
    def _format_response_detail(cls, response: requests.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            response_text = getattr(response, "text", "") or ""
            return response_text.strip()[:500] or "no response body"
        if isinstance(payload, dict):
            return cls._format_payload(payload)
        return repr(payload)[:500]

    def _json_response(self, response: requests.Response, *, action: str) -> dict:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            status_code = getattr(response, "status_code", "unknown")
            detail = self._format_response_detail(response)
            raise CompressionUnavailable(
                f"Vidaio {action} failed with HTTP {status_code}: {detail}"
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise CompressionUnavailable(
                f"Vidaio {action} response was not valid JSON."
            ) from exc
        if not isinstance(payload, dict):
            raise CompressionUnavailable(
                f"Vidaio {action} response was not a JSON object."
            )
        return payload

    @staticmethod
    def _is_rate_limited(response: requests.Response) -> bool:
        return getattr(response, "status_code", None) == 429

    def _retry_after_seconds(self, response: requests.Response) -> float:
        headers = getattr(response, "headers", {}) or {}
        raw_value = headers.get("Retry-After") if hasattr(headers, "get") else None
        try:
            retry_after = float(raw_value) if raw_value else 60.0
        except (TypeError, ValueError):
            retry_after = 60.0
        return max(self.poll_interval_seconds, retry_after)

    @staticmethod
    def _sleep_until(deadline: float, delay_seconds: float) -> bool:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(float(delay_seconds), remaining))
        return True

    def _start(self, video_url: str) -> str:
        response = requests.post(
            f"{self.api_base_url}/compression",
            headers=self._headers(),
            json={
                "video_url": video_url,
                "compression_type": self.compression_type,
                "tier": self.tier,
            },
            timeout=self.timeout_seconds,
        )
        payload = self._json_response(response, action="start")
        task_id = payload.get("task_id")
        if not task_id:
            raise CompressionUnavailable("Vidaio start response did not include task_id.")
        return str(task_id)

    def _wait_until_completed(self, *, task_id: str, deadline: float) -> None:
        next_delay = self.poll_interval_seconds
        while True:
            if not self._sleep_until(deadline, next_delay):
                raise CompressionUnavailable("Vidaio compression timed out.")
            response = requests.get(
                f"{self.api_base_url}/compression/{task_id}",
                headers=self._headers(),
                timeout=self.timeout_seconds,
            )
            if self._is_rate_limited(response):
                next_delay = self._retry_after_seconds(response)
                continue

            payload = self._json_response(response, action="status")
            status = str(payload.get("status") or "").lower()
            if status == "completed":
                return
            if status in self.TERMINAL_FAILURES:
                raise CompressionUnavailable(
                    f"Vidaio compression failed: {self._format_payload(payload)}"
                )
            next_delay = self.poll_interval_seconds

    def _result_url(self, *, task_id: str, deadline: float) -> str:
        next_delay = 0.0
        while True:
            if next_delay and not self._sleep_until(deadline, next_delay):
                raise CompressionUnavailable("Vidaio compression timed out.")
            response = requests.get(
                f"{self.api_base_url}/compression/{task_id}/result",
                headers=self._headers(),
                timeout=self.timeout_seconds,
            )
            if not self._is_rate_limited(response):
                break
            next_delay = self._retry_after_seconds(response)

        payload = self._json_response(response, action="result")
        result_url = (
            payload.get("download_url")
            or payload.get("url")
            or payload.get("output_url")
            or payload.get("video_url")
        )
        if not result_url:
            raise CompressionUnavailable("Vidaio result did not include a download URL.")
        return str(result_url)

    def compress(self, *, video_url: str) -> str:
        if not self.api_base_url:
            raise CompressionUnavailable("Vidaio compression API URL is not configured.")

        task_id = self._start(video_url)
        deadline = time.monotonic() + self.timeout_seconds
        self._wait_until_completed(task_id=task_id, deadline=deadline)
        return self._result_url(task_id=task_id, deadline=deadline)

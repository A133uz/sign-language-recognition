"""Test link selection and failure handling without using the network."""

from io import BytesIO
from urllib.error import HTTPError, URLError

import pytest

from sign_language import video_probe
from sign_language.cli import main


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://example.com/video.MP4?version=1", True),
        ("https://youtube.com/watch?v=123", False),
        ("file:///video.mp4", False),
        ("https://user:pass@example.com/video.mp4", False),
        (None, False),
    ],
)
def test_direct_mp4_selection(url, expected):
    assert video_probe.is_direct_mp4(url) is expected


def test_probe_gloss_skips_pages_and_respects_limit(monkeypatch):
    sign = {
        "gloss": "hello",
        "instances": [
            {"video_id": "page", "url": "https://youtube.com/watch?v=1"},
            {"video_id": "first", "url": "https://example.com/1.mp4"},
            {"video_id": "second", "url": "https://example.com/2.mp4"},
        ],
    }
    monkeypatch.setattr(video_probe, "probe_video", lambda item: {"video_id": item["video_id"]})
    report = video_probe.probe_gloss(sign, limit=1)
    assert report["direct_mp4_candidates"] == 2
    assert report["tested"] == 1
    assert report["results"] == [{"video_id": "first"}]


def test_probe_decodes_downloaded_data(monkeypatch):
    class Response(BytesIO):
        headers = {"Content-Length": "5"}

    monkeypatch.setattr(video_probe, "urlopen", lambda *args, **kwargs: Response(b"video"))
    monkeypatch.setattr(video_probe, "decode_first_frame", lambda data: (320, 240))
    result = video_probe.probe_video({"video_id": "1", "url": "https://example.com/a.mp4"})
    assert result["status"] == "decoded"
    assert (result["bytes"], result["width"], result["height"]) == (5, 320, 240)


def test_probe_reports_http_and_network_errors(monkeypatch):
    def fail_http(*args, **kwargs):
        raise HTTPError("https://example.com/a.mp4", 404, "Not found", {}, None)

    instance = {"url": "https://example.com/a.mp4"}
    monkeypatch.setattr(video_probe, "urlopen", fail_http)
    assert video_probe.probe_video(instance)["http_status"] == 404

    def fail_network(*args, **kwargs):
        raise URLError("network unavailable")

    monkeypatch.setattr(video_probe, "urlopen", fail_network)
    assert video_probe.probe_video(instance)["status"] == "request_error"


def test_probe_stops_before_decoding_oversized_response(monkeypatch):
    class Response(BytesIO):
        headers = {"Content-Length": str(video_probe.MAX_VIDEO_BYTES + 1)}

    monkeypatch.setattr(video_probe, "urlopen", lambda *args, **kwargs: Response(b""))
    result = video_probe.probe_video({"url": "https://example.com/a.mp4"})
    assert result["status"] == "too_large"


def test_probe_detects_oversized_content_without_length_header(monkeypatch):
    class Response(BytesIO):
        headers = {}

    monkeypatch.setattr(video_probe, "MAX_VIDEO_BYTES", 4)
    monkeypatch.setattr(video_probe, "urlopen", lambda *args, **kwargs: Response(b"12345"))
    result = video_probe.probe_video({"url": "https://example.com/a.mp4"})
    assert result["status"] == "too_large"


def test_probe_reports_invalid_video_data(monkeypatch):
    class Response(BytesIO):
        headers = {}

    monkeypatch.setattr(video_probe, "urlopen", lambda *args, **kwargs: Response(b"not video"))
    monkeypatch.setattr(video_probe, "decode_first_frame", lambda data: None)
    result = video_probe.probe_video({"url": "https://example.com/a.mp4"})
    assert result["status"] == "decode_failed"


def test_probe_rejects_unbounded_limit(capsys):
    with pytest.raises(SystemExit) as error:
        main(["probe-wlasl", "hello", "--limit", "11"])
    assert error.value.code == 1
    assert "between 1 and 10" in capsys.readouterr().err

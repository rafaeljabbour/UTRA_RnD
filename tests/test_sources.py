import pytest

from utra_tracker.sources import Source, SourceError, parse_source


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0, Source("camera", 0)),
        (2, Source("camera", 2)),
        ("1", Source("camera", 1)),
        (" 3 ", Source("camera", 3)),
        ("videos/maze.mp4", Source("file", "videos/maze.mp4")),
        (r"C:\videos\maze run.mp4", Source("file", r"C:\videos\maze run.mp4")),
        ("rtsp://10.0.0.5:554/stream1", Source("stream", "rtsp://10.0.0.5:554/stream1")),
        ("RTSP://cam/live", Source("stream", "RTSP://cam/live")),
        ("http://192.168.1.20:8080/video", Source("stream", "http://192.168.1.20:8080/video")),
        ("https://example.com/feed.m3u8", Source("stream", "https://example.com/feed.m3u8")),
    ],
)
def test_parse_source(raw, expected):
    assert parse_source(raw) == expected


@pytest.mark.parametrize("raw", [-1, "", "   ", True])
def test_parse_source_rejects(raw):
    with pytest.raises(SourceError):
        parse_source(raw)


def test_live_flag():
    assert parse_source(0).is_live
    assert parse_source("rtsp://x").is_live
    assert not parse_source("a.mp4").is_live

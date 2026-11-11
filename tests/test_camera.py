import pytest

from invoice_parser.camera import CameraError, ip_camera_snapshot_url


@pytest.mark.parametrize("given, expected", [
    ("http://192.168.1.20:8080", "http://192.168.1.20:8080/shot.jpg"),
    ("http://192.168.1.20:8080/", "http://192.168.1.20:8080/shot.jpg"),
    ("http://192.168.1.20:8080/video", "http://192.168.1.20:8080/shot.jpg"),
    ("http://192.168.1.20:8080/shot.jpg", "http://192.168.1.20:8080/shot.jpg"),
    ("http://10.0.0.7:4747", "http://10.0.0.7:4747/shot.jpg"),
])
def test_snapshot_url(given, expected):
    assert ip_camera_snapshot_url(given) == expected


@pytest.mark.parametrize("bad", ["http://8.8.8.8:8080", "ftp://192.168.1.2/x", "192.168.1.20:8080", ""])
def test_rejects_public_or_malformed_addresses(bad):
    with pytest.raises(CameraError):
        ip_camera_snapshot_url(bad)

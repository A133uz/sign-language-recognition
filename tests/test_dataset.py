import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from sign_language import dataset


class DatasetTests(unittest.TestCase):
    def test_archive_and_csv_paths_cannot_escape_their_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("../outside", "/outside", "C:/outside", "a\\..\\outside"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    dataset.safe_member(root, name)
            self.assertEqual(dataset.safe_member(root, "a/video.mp4"), root / "a/video.mp4")

    def test_resumed_download_and_rejection_of_a_corrupt_prefix(self):
        content = b"PK official archive fixture" * 100

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass

            def do_HEAD(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(content)))
                self.send_header("ETag", '"fixture-v1"')
                self.end_headers()

            def do_GET(self):
                start, end = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
                data = content[start:end + 1]
                self.send_response(206)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(content)}")
                self.send_header("ETag", '"fixture-v1"')
                self.end_headers()
                self.wfile.write(data)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                archive = Path(directory) / "dataset.zip"
                partial = archive.with_suffix(".zip.part")
                partial.write_bytes(content[:33])
                with patch.object(dataset, "URL", f"http://127.0.0.1:{server.server_port}/dataset.zip"), \
                     patch.object(dataset, "ARCHIVE", archive):
                    dataset.download_dataset(workers=2)
                    self.assertEqual(archive.read_bytes(), content)
                    archive.unlink()
                    partial.write_bytes(b"incorrect partial")
                    with self.assertRaisesRegex(ValueError, "does not match"):
                        dataset.download_dataset(workers=2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

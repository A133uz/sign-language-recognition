"""In-memory, user-controlled preview frames for visual recognition diagnostics."""

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from sign_language.landmarks import HandExtractor


def review_gestures(
    camera: int = 0, *, review_endpoint: str | None = None, tracking: bool = False
) -> None:
    """Show three body/hand views per gesture; never save images or training samples."""
    import cv2

    if review_endpoint:
        endpoint = urllib.parse.urlsplit(review_endpoint)
        if endpoint.scheme != "http" or endpoint.hostname != "127.0.0.1":
            raise ValueError("Direct preview output must use a local loopback endpoint")

    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("Cannot open webcam")
    title = "Gesture review"
    prompts = ["ME (pronoun)", "THANK YOU"]
    index, stage, deadline, started = 0, "ready", 0.0, 0.0
    snapshots, board = [], None
    try:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        with HandExtractor(
            Path("assets/models/hand_landmarker.task"),
            Path("assets/models/pose_landmarker_lite.task") if tracking else None,
            include_world=tracking,
        ) as extractor:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError("Webcam stopped returning frames")
                now = time.monotonic()
                if stage == "countdown" and now >= deadline:
                    stage, started, snapshots = "capture", now, []
                if stage == "capture":
                    targets = (0.4, 0.9, 1.4)
                    if len(snapshots) < 3 and now - started >= targets[len(snapshots)]:
                        snapshots.append(frame.copy())
                    if len(snapshots) == 3:
                        board = np.zeros((880 if tracking else 660, 960, 3), dtype=np.uint8)
                        cv2.putText(
                            board,
                            prompts[index] + " - three preview frames (mirrored)",
                            (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            board,
                            "Wait for review. N: next sign | Q: close. Images stay in memory.",
                            (10, board.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 255, 0),
                            1,
                        )
                        for column, shot in enumerate(snapshots):
                            h, w = shot.shape[:2]
                            model_shot = shot
                            if tracking and max(h, w) > 480:
                                scale = 480 / max(h, w)
                                model_shot = cv2.resize(shot, (round(w * scale), round(h * scale)))
                            features = extractor.extract(model_shot)
                            annotated = cv2.flip(shot, 1)
                            if tracking:
                                from sign_language.tracking_preview import annotate_hands

                                annotated = annotate_hands(shot, features, mirrored=True)
                            board[40:280, column * 320 : (column + 1) * 320] = cv2.resize(
                                annotated, (320, 240)
                            )
                            choices = []
                            for slot in range(2):
                                if not features[slot * 64 + 63]:
                                    continue
                                points = features[slot * 64 : slot * 64 + 63].reshape(21, 3)[
                                    :, :2
                                ] * [w, h]
                                low, high = points.min(0), points.max(0)
                                choices.append((float(np.prod(high - low)), low, high, slot))
                            if choices:
                                _, low, high, slot = max(choices, key=lambda value: value[0])
                                center = (low + high) / 2
                                side = max(100, int(max(high - low) * 1.8))
                                left, top = (
                                    max(0, int(center[0] - side / 2)),
                                    max(0, int(center[1] - side / 2)),
                                )
                                right, bottom = (
                                    min(w, int(center[0] + side / 2)),
                                    min(h, int(center[1] + side / 2)),
                                )
                                if right > left and bottom > top:
                                    board[300:620, column * 320 : (column + 1) * 320] = cv2.resize(
                                        annotated[top:bottom, w - right : w - left], (320, 320)
                                    )
                                if tracking:
                                    from sign_language.tracking_preview import estimated_hand_views

                                    board[640:800, column * 320 : (column + 1) * 320] = cv2.resize(
                                        estimated_hand_views(features, slot), (320, 160)
                                    )
                        if tracking:
                            cv2.putText(
                                board,
                                "Tips: thumb 4 / index 8 / middle 12 / ring 16 / little 20",
                                (10, 825),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                (230, 230, 230),
                                1,
                            )
                            cv2.putText(
                                board,
                                "Body/crops mirrored; world views unmirrored. Not measured depth.",
                                (10, 846),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (230, 230, 230),
                                1,
                            )
                        stage = "result"
                        if review_endpoint:
                            encoded_ok, encoded = cv2.imencode(
                                ".jpg", board, [cv2.IMWRITE_JPEG_QUALITY, 85]
                            )
                            if not encoded_ok:
                                raise RuntimeError("Cannot encode preview image")
                            request = urllib.request.Request(
                                review_endpoint,
                                data=encoded.tobytes(),
                                method="POST",
                                headers={"Content-Type": "image/jpeg", "X-Gesture": prompts[index]},
                            )
                            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                            with opener.open(request, timeout=10) as response:
                                if response.status != 200:
                                    raise RuntimeError("Local preview receiver rejected the image")
                            print(json.dumps({"preview_ready": prompts[index]}), flush=True)
                        cv2.setWindowTitle(
                            title, "Gesture review - " + prompts[index] + " captured"
                        )
                if stage == "result":
                    display = board
                else:
                    display = cv2.flip(frame, 1)
                    if stage == "ready":
                        lines = [
                            prompts[index],
                            "SPACE: 2-second countdown, then show the sign once",
                        ]
                    elif stage == "countdown":
                        lines = [prompts[index], f"Get ready: {max(0, deadline - now):.1f}s"]
                    else:
                        lines = [prompts[index], "SHOW SIGN NOW - three frames for review"]
                    for row, line in enumerate(lines):
                        cv2.putText(
                            display,
                            line,
                            (10, 30 + row * 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 255, 0),
                            2,
                        )
                cv2.imshow(title, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord(" ") and stage == "ready":
                    stage, deadline = "countdown", time.monotonic() + 2
                if key == ord("n") and stage == "result":
                    index += 1
                    snapshots, board = [], None
                    if index == len(prompts):
                        break
                    stage = "ready"
                    cv2.setWindowTitle(title, "Gesture review - " + prompts[index])
    finally:
        capture.release()
        cv2.destroyAllWindows()

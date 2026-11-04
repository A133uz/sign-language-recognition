"""In-memory visual diagnostics for the exact hand features given to a classifier."""

import numpy as np

# BGR colors, one per thumb/index/middle/ring/little finger.
FINGER_COLORS = ((60, 60, 255), (40, 255, 40), (255, 220, 0), (255, 80, 200), (0, 180, 255))
FINGER_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)


def annotate_hands(
    frame: np.ndarray, features: np.ndarray, *, mirrored: bool = False
) -> np.ndarray:
    """Return an annotated copy; do not write images or alter classifier features."""
    import cv2

    features = np.asarray(features)
    if frame.ndim != 3 or frame.shape[2] != 3 or features.shape not in ((128,), (160,), (288,)):
        raise ValueError("Tracking preview requires a color frame and supported feature vector")
    if not np.isfinite(features).all():
        raise ValueError("Tracking features must be finite")
    result = cv2.flip(frame, 1) if mirrored else frame.copy()
    height, width = frame.shape[:2]
    for slot in range(2):
        if not features[slot * 64 + 63]:
            continue
        points = features[slot * 64 : slot * 64 + 63].reshape(21, 3)[:, :2]
        pixels = np.rint(points * [width, height]).astype(int)
        if mirrored:
            pixels[:, 0] = width - 1 - pixels[:, 0]
        for chain, color in zip(FINGER_CHAINS, FINGER_COLORS, strict=True):
            for left, right in zip(chain[:-1], chain[1:], strict=True):
                cv2.line(result, tuple(pixels[left]), tuple(pixels[right]), color, 1, cv2.LINE_AA)
            for index in chain[1:]:
                cv2.circle(result, tuple(pixels[index]), 2, color, -1, cv2.LINE_AA)
            tip = chain[-1]
            cv2.putText(
                result,
                str(tip),
                tuple(pixels[tip] + [3, -3]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )
    return result


def estimated_hand_views(features: np.ndarray, slot: int, size: int = 240) -> np.ndarray:
    """Draw estimated normalized world hands in XY and XZ; these are not measured depth."""
    import cv2

    features = np.asarray(features)
    if features.shape != (288,) or slot not in (0, 1) or not 100 <= size <= 600:
        raise ValueError("World-hand views require world features, a hand slot and valid size")
    if not np.isfinite(features).all():
        raise ValueError("Tracking features must be finite")
    result = np.zeros((size, size * 2, 3), dtype=np.uint8)
    hand = features[160 + slot * 64 : 224 + slot * 64]
    if not hand[63]:
        cv2.putText(
            result,
            "No estimated world hand",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        return result
    points = hand[:63].reshape(21, 3)
    scale = (size - 60) / (2 * max(1, float(np.abs(points).max())))
    for column, axes in enumerate(((0, 1), (0, 2))):
        pixels = np.rint(points[:, axes] * scale + [size / 2 + column * size, size / 2]).astype(int)
        cv2.putText(
            result,
            "estimated " + ("XY" if column == 0 else "XZ"),
            (column * size + 10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
        )
        for chain, color in zip(FINGER_CHAINS, FINGER_COLORS, strict=True):
            for left, right in zip(chain[:-1], chain[1:], strict=True):
                cv2.line(result, tuple(pixels[left]), tuple(pixels[right]), color, 2, cv2.LINE_AA)
            cv2.putText(
                result,
                str(chain[-1]),
                tuple(pixels[chain[-1]]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
    return result

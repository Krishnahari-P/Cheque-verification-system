
import cv2
import numpy as np

def resize_with_padding(img, target_size=256, pad_value=255):
    h, w = img.shape[:2]
    scale = target_size / max(h, w)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.full(
        (target_size, target_size),
        pad_value,
        dtype=resized.dtype
    )

    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2

    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas
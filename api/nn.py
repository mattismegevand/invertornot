from typing import Union

import numpy as np
import onnxruntime
from PIL.Image import Image


def resize(image: Image, size: Union[int, tuple]) -> Image:
    if isinstance(size, int):
        w, h = image.size
        if h > w:
            new_h = int(size * h / w)
            size = (size, new_h)
        else:
            new_w = int(size * w / h)
            size = (new_w, size)
    return image.resize(size)


def center_crop(img: Image, size: tuple = (128, 128)) -> Image:
    w, h = img.size
    left = (w - size[0]) / 2
    top = (h - size[1]) / 2
    right = (w + size[0]) / 2
    bottom = (h + size[1]) / 2
    return img.crop((left, top, right, bottom))


def img_to_np(img: Image) -> np.ndarray:
    arr = np.array(img) / 255.0
    arr = arr.transpose((2, 0, 1))
    return arr.astype(np.float32)


class NN:
    def __init__(self, model_path: str) -> None:
        self.sess = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

    def pred(self, img: Image) -> int:
        x = self.transform(img)[np.newaxis, :]
        inp = {self.sess.get_inputs()[0].name: x}
        out = self.sess.run(None, inp)[0].item()
        return int(out >= 0.5)

    def transform(self, img: Image) -> np.ndarray:
        img = resize(img, 128)
        img = center_crop(img, (128, 128))
        return img_to_np(img)

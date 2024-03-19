import onnxruntime
import torch
from PIL.Image import Image
from torchvision.transforms import v2


class NN:
  def __init__(self, model_path: str) -> None:
    self.transform = v2.Compose([
      v2.ToImage(),
      v2.Resize(128),
      v2.CenterCrop(128),
      v2.ToDtype(torch.float32, scale=True),
    ])
    self.sess = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])

  def pred(self, img: Image) -> int:
      x = self.transform(img).unsqueeze(0)
      inp = {self.sess.get_inputs()[0].name: x.detach().cpu().numpy()}
      out = self.sess.run(None, inp)[0].item()
      return int(out >= 0.5)
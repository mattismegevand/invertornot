#!/usr/bin/env python

import numpy as np
import onnx
import onnxruntime
import timm
import torch
import torch.nn as nn


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == "__main__":
    print("Converting PyTorch model to ONNX format...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("efficientnet_b0", pretrained=True).to(device)
    model.classifier = nn.Sequential(nn.Linear(model.classifier.in_features, 1), nn.Sigmoid()).to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    batch_size = 1
    x = torch.randn(batch_size, 3, 128, 128, requires_grad=True)
    torch_out = model(x)

    torch.onnx.export(
        model,
        x,
        "model.onnx",
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    onnx_model = onnx.load("model.onnx")
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Done.")

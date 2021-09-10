import torch
import torchvision.models as models
import onnxruntime as onnxrt
import numpy as np
print(f"torch: {torch.__version__}")

# model = models.resnet50(pretrained=True)
#
# model.eval()
#
# dummy_input = torch.randn(1, 3, 224, 224)
#
# input_names = ["actual_input"]
# output_names = ["output"]
#
# torch.onnx.export(model,
#                   dummy_input,
#                   "resnet50.onnx",
#                   verbose=False,
#                   input_names=input_names,
#                   output_names=output_names,
#                   export_params=True,
#                   )
#
# onnx_session = onnxrt.InferenceSession("resnet50.onnx")
# onnx_inputs = {onnx_session.get_inputs()[0].name: np.to_numpy(img)}
# onnx_output = onnx_session.run(None, onnx_inputs)
# img_label = onnx_outputort_outs[0]

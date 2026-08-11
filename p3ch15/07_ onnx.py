import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 128, 5)

    def forward(self, x):
        return torch.relu(self.conv1(x))

input_tensor = torch.rand((1, 1, 128, 128), dtype=torch.float32)

model = MyModel()

"""
为了将模型导出到 ONNX, 我们需要运行一个带有虚拟输入的模型: 输入张量的值实际上并不重要, 重要的是它们的形状和类型是否正确.
通过调用 torch.onnx.export() 函数, PyTorch 将跟踪模型执行的计算. 并用提供的名称将它们序列化为 ONNX 文件.
"""
torch.onnx.export(
    model,                  # model to export
    (input_tensor,),        # inputs of the model,
    "my_model.onnx",        # filename of the ONNX model
    input_names=["input"],  # Rename inputs for the ONNX model
    dynamo=True             # True or False to select the exporter to use
)

#####################################################

# Export the model using torch.onnx.export
torch.onnx.export(
    model,                              # model being run
    torch.randn(1, 28, 28).to(device),  # model input (or a tuple for multiple inputs)
    "fashion_mnist_model.onnx",         # where to save the model (can be a file or file-like object)
    input_names = ['input'],            # the model's input names
    output_names = ['output']           # the model's output names
)

# Load the onnx model with onnx.load
import onnx
onnx_model = onnx.load("fashion_mnist_model.onnx")
onnx.checker.check_model(onnx_model)

# Create inference session using ort.InferenceSession
import onnxruntime as ort
import numpy as np
x, y = test_data[0][0], test_data[0][1]
ort_sess = ort.InferenceSession('fashion_mnist_model.onnx')
outputs = ort_sess.run(None, {'input': x.numpy()})

# Print Result
predicted, actual = classes[outputs[0][0].argmax(0)], classes[y]
print(f'Predicted: "{predicted}", Actual: "{actual}"')

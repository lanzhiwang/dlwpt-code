import requests
import json
import io
import torch

im, cl, id, pos = torch.load("data/p3ch15/cls_val_example.pt")
"""
print("im:", im)
print("im:", im.shape)
print("---" * 8)
print("cl:", cl)
print("cl:", cl.shape)
print("---" * 8)
print("id:", id)
print("id:", type(id))
print("---" * 8)
print("pos:", pos)
print("pos:", pos.shape)

im: tensor([[[[  56.,   66.,  ...,   11.,  -21.],
          [  39.,   62.,  ...,    5.,   12.],
          ...,
          [-935., -946.,  ...,  217.,  216.],
          [-915., -914.,  ...,  209.,  246.]],

         [[  10.,   23.,  ...,   13.,  -14.],
          [  -8.,   16.,  ...,   -9.,    7.],
          ...,
          [-912., -945.,  ...,  237.,  224.],
          [-910., -923.,  ...,  225.,  239.]],

         ...,

         [[-959., -916.,  ..., -991., -967.],
          [-927., -863.,  ..., -983., -969.],
          ...,
          [-905., -896.,  ...,  327.,  374.],
          [-896., -896.,  ...,  372.,  387.]],

         [[-944., -909.,  ..., -993., -961.],
          [-909., -843.,  ..., -987., -977.],
          ...,
          [-898., -898.,  ...,  266.,  275.],
          [-883., -875.,  ...,  292.,  283.]]]])
im: torch.Size([1, 32, 48, 48])
------------------------
cl: tensor([1, 0])
cl: torch.Size([2])
------------------------
id: 1.3.6.1.4.1.14519.5.2.1.6279.6001.922852847124879997825997808179
id: <class 'str'>
------------------------
pos: tensor([341.0240, 284.9991, 220.0178])
pos: torch.Size([3])

list(im.shape)
[1, 32, 48, 48]

json.dumps({"shape": list(im.shape)})
{"shape": [1, 32, 48, 48]}

type(meta)
_io.StringIO

im.numpy().shape, type(im.numpy())
((1, 32, 48, 48), numpy.ndarray)

type(bytearray(im.numpy()))
bytearray

blob = bytearray(im.numpy())
in_tensor = torch.from_numpy(np.frombuffer(blob, dtype=np.float32))
in_tensor.shape
torch.Size([73728])

"""

meta = io.StringIO(json.dumps({"shape": list(im.shape)}))
data = io.BytesIO(bytearray(im.numpy()))
r = requests.post("http://localhost:8000/predict", files={"meta": meta, "blob": data})
response = json.loads(r.content)

print("Model predicted probability of being maignant:", response["prob_malignant"])

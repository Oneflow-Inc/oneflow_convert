"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import tempfile
import oneflow as flow
from oneflow_onnx.oneflow2onnx.util import convert_to_onnx_and_check


class AmpIdentity(flow.nn.Module):
    def __init__(self) -> None:
        super(AmpIdentity, self).__init__()
        self.matmul = flow.nn.Linear(20, 30)

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        x = self.matmul(x)
        x = flow._C.amp_white_identity(x)
        x = flow._C.amp_black_identity(x)
        return x.mean()


class AmpIdentityGraph(flow.nn.Graph):
    def __init__(self, model):
        super().__init__()
        self.m = model
        self.config.proto.prune_amp_white_identity_ops = False

    def build(self, x):
        out = self.m(x)
        return out


def test_amp_identity():
    amp_id_module = AmpIdentity()
    amp_id_module = amp_id_module.to("cuda")

    graph = AmpIdentityGraph(amp_id_module)
    graph._compile(flow.randn(1, 20).to("cuda"))

    with tempfile.TemporaryDirectory() as tmpdirname:
        flow.save(amp_id_module.state_dict(), tmpdirname, save_as_external_data=True)
        convert_to_onnx_and_check(graph, onnx_model_path="/tmp", device="gpu")


if __name__ == "__main__":
    test_amp_identity()

import tempfile
import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F
from oneflow.fx.passes.quantization import quantization_aware_training
from oneflow.fx.passes.dequantization import dequantization_aware_training
from oneflow_onnx.oneflow2onnx.util import convert_to_onnx_and_check

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

resnet18 = ResNet18()
resnet18 = resnet18.to("cuda")
resnet18.eval()

gm: flow.fx.GraphModule = flow.fx.symbolic_trace(resnet18)
qconfig = {
    'quantization_bit': 8, 
    'quantization_scheme': "symmetric", 
    'quantization_formula': "google", 
    'per_layer_quantization': True,
    'momentum': 0.95,
}

quantization_resnet18 = quantization_aware_training(gm, flow.randn(1, 3, 32, 32).to("cuda"), qconfig)
quantization_resnet18 = quantization_resnet18.to("cuda")
quantization_resnet18.eval()

origin_gm: flow.fx.GraphModule = flow.fx.symbolic_trace(resnet18)
dequantization_resnet18 = dequantization_aware_training(origin_gm, gm, flow.randn(1, 3, 32, 32).to("cuda"), qconfig)
dequantization_resnet18 = dequantization_resnet18.to("cuda")
dequantization_resnet18.eval()
print(dequantization_resnet18)

class ResNet18Graph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.m = dequantization_resnet18

    def build(self, x):
        out = self.m(x)
        return out

def test_resnet():
    
    resnet_graph = ResNet18Graph()
    resnet_graph._compile(flow.randn(1, 3, 32, 32).to("cuda"))
    # print(resnet_graph)    

    with tempfile.TemporaryDirectory() as tmpdirname:
        flow.save(dequantization_resnet18.state_dict(), tmpdirname)
        convert_to_onnx_and_check(resnet_graph, flow_weight_dir=tmpdirname, onnx_model_path="/tmp", print_outlier=True)

test_resnet()

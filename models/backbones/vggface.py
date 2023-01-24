import torch
from torchvision.models import ResNet
from torchvision.models.resnet import Bottleneck

import config
import pickle


class VGGFace(ResNet):

    def __init__(self ,*args, **kwargs):
        self.output_dim = 0
        super().__init__(*args, **kwargs)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        res = super()._make_layer(block, planes, blocks, stride=stride, dilate=dilate)
        self.output_dim += self.inplanes
        return res

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        features = []
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)

        return features

    @property
    def features_dim(self):
        return self.output_dim
    
def resnet50(**kwargs):
    return VGGFace(Bottleneck, [3, 4, 6, 3],**kwargs)


def vggface():
    model = resnet50(num_classes=8631)
    with open(config.PRETRAINED_WEIGHTS_PATH["vggface"], 'rb') as f:
        obj = f.read()
    weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
    model.load_state_dict(weights)
    # model.eval()
    return model

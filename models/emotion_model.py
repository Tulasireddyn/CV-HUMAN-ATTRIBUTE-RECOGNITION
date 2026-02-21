import torch
import torch.nn as nn
import yaml
from torchvision.models import (
    resnet18,
    resnet50,
    ResNet18_Weights,
    ResNet50_Weights
)


class EmotionModel(nn.Module):
    def __init__(self, model_name="resnet18", num_classes=7, pretrained=True):
        super(EmotionModel, self).__init__()

        if model_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.model = resnet18(weights=weights)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)

        elif model_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.model = resnet50(weights=weights)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)

        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def forward(self, x):
        return self.model(x)


def load_model_from_config(config_path: str):
    """Load model based on YAML config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_name = config.get("model_name", "resnet18")
    num_classes = config.get("num_classes", 7)
    pretrained = config.get("pretrained", True)

    return EmotionModel(model_name=model_name,
                        num_classes=num_classes,
                        pretrained=pretrained)

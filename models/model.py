import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class AgeGenderModel(nn.Module):
    def __init__(self, pretrained=True, age_hidden=128, gender_classes=2):
        super().__init__()
        
        # Load ResNet18 backbone
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base = resnet18(weights=weights)
        
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # remove FC layer
        num_features = base.fc.in_features

        # Age regression head
        self.age_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, age_hidden),
            nn.ReLU(),
            nn.Linear(age_hidden, 1)
        )

        # Gender classification head
        self.gender_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, gender_classes)
        )

    def forward(self, x):
        features = self.backbone(x)  # output: [batch, num_features, 1, 1]
        age_out = self.age_head(features).squeeze(1)  # regression output
        gender_out = self.gender_head(features)       # classification logits
        return age_out, gender_out

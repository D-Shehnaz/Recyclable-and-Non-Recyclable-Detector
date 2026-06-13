import torch.nn as nn
from torchvision import models

def get_model(num_classes=6):
    model = models.mobilenet_v2(weights=None)

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, num_classes)
    )

    return model
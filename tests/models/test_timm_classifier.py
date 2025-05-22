import pytest
import torch
import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.timm_classifier import TimmClassifier

def test_timm_classifier_forward(device):
    model = TimmClassifier(base_model='resnet18', num_classes=2)
    model.to(device)
    model.eval()

    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    
    with torch.no_grad():
        output = model(x)
    
    assert output.shape == (batch_size, 2)
    assert not torch.isnan(output).any()

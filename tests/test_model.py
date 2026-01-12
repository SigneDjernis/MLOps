from signe_proj.model import MyAwesomeModel
import torch

def test_model():
    model = MyAwesomeModel()
    x = torch.randn(1, 1, 28, 28)
    x_flat = x.view(x.size(0), -1)  # flatten the images
    y = model(x_flat)
    assert y.shape == (1, 10)
from torch import nn
import torch


class MyAwesomeModel(nn.Module):
    """Basic Neural Network with 1 hidden layer."""

    def __init__(self) -> None:
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each class
        self.output = nn.Linear(256, 10)

        # Define sigmoid activation and softmax output
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """Forward pass through the network, returns the output logits."""
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        return self.softmax(x)


if __name__ == "__main__":
    model = MyAwesomeModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    dummy_input = dummy_input.view(-1, 784)  # Flatten the input
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

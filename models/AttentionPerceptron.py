from torch import nn

class AttentionPerceptron(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()

        # TODO: Please design your CNN layers. They are typically made of some Conv2d layers and pool layers
        self.fc_layers = nn.Sequential(
            nn.MultiheadAttention(8, 1),
            nn.Linear(8, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.out_layers = nn.Sequential(
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        
        x = self.fc_layers(x)
        
        return self.out_layers(x)
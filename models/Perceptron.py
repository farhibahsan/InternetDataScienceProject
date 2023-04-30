from torch import nn
import torch.nn.functional as F

class Perceptron(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()

        # TODO: Please design your CNN layers. They are typically made of some Conv2d layers and pool layers
        # self.fc_layers = nn.Sequential(
        #     nn.Linear(8, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU()
        # )

        # self.out_layers = nn.Sequential(
        #     nn.Linear(hidden_size, num_classes),
        #     nn.Sigmoid()
        # )

        self.l1 = nn.Linear(8, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        
        return F.sigmoid(self.l4(x))
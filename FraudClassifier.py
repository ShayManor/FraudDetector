from torch import nn


class FraudClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 256)
        self.r1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.r2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 256)
        self.r3 = nn.ReLU()
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.r1(x)
        x = self.fc2(x)
        x = self.r2(x)
        x = self.fc3(x)
        x = self.r3(x)
        x = self.fc4(x)
        return x
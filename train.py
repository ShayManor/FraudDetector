import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from FraudClassifier import FraudClassifier
from FraudDataset import FraudDataset
# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
if __name__ == '__main__':
    model = FraudClassifier()
    train = FraudDataset('train.csv')
    finetune = FraudDataset('finetune.csv')
    train_dataloader = DataLoader(batch_size=16, dataset=train, shuffle=True)
    ft_dataloader = DataLoader(batch_size=4, dataset=train, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    initial_epochs = 30
    ft_epochs = 10
    for epoch in range(initial_epochs):
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
        print("--------------------------------")
    torch.save(model.state_dict(), 'weights.pt')

    for epoch in range(ft_epochs):
        for inputs, labels in ft_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
        print("--------------------------------")
    torch.save(model.state_dict(), 'ft_weights.pt')
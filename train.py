import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from FraudClassifier import FraudClassifier
from FraudDataset import FraudDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    model = FraudClassifier()
    model.to(device)
    train = FraudDataset('train.csv')
    finetune = FraudDataset('finetune.csv')
    train_dataloader = DataLoader(batch_size=256, dataset=train, shuffle=True, num_workers=8, prefetch_factor=4, persistent_workers=True, pin_memory=True)
    ft_dataloader = DataLoader(batch_size=32, dataset=finetune, shuffle=True, num_workers=8, prefetch_factor=4, persistent_workers=True, pin_memory=True)
    df = train.get_df()
    N_neg, N_pos = df['isFraud'].value_counts()[0], df['isFraud'].value_counts()[1]
    pos_weight = torch.tensor([N_neg / N_pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    initial_epochs = 30
    ft_epochs = 10
    start_time = time.time()
    for epoch in range(initial_epochs):
        for inputs, labels in train_dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().unsqueeze(1)
            if torch.isnan(inputs).any():
                raise RuntimeError("🛑 NaN in inputs")
            if torch.isnan(labels).any():
                raise RuntimeError("🛑 NaN in labels")
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}, Time:{time.time() - start_time}")
        start_time = 0
        print("--------------------------------")
    torch.save(model.state_dict(), 'weights.pt')
    df = finetune.get_df()
    N_neg, N_pos = df['isFraud'].value_counts()[0], df['isFraud'].value_counts()[1]
    pos_weight = torch.tensor([N_neg / N_pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for epoch in range(ft_epochs):
        for inputs, labels in ft_dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}, Time:{time.time() - start_time}")
        start_time = 0
        print("--------------------------------")
    torch.save(model.state_dict(), 'ft_weights.pt')
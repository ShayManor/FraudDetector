import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import float32
from torch.utils.data import Dataset

def parseType(type: str):
    if type == 'PAYMENT':
        return 0
    if type == 'TRANSFER':
        return 1
    if type == 'CASH_OUT':
        return -1
    if type == 'DEBIT':
        return 2

def parseName(name: str):
    if name[0] == 'C':
        return 0
    if name[0] == 'M':
        return 1
    return -1


class FraudDataset(Dataset):
    def __init__(self, path):
        self.df = pd.read_csv(path, header=0)
        scalar = MinMaxScaler()
        self.df['nameOrig'] = self.df['nameOrig'].apply(parseName)
        self.df['nameDest'] = self.df['nameDest'].apply(parseName)
        self.df['type'] = self.df['type'].apply(parseType)
        self.features = [
            'step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest',
            'newbalanceDest',
        ]
        self.df[self.features] = scalar.fit_transform(self.df[self.features])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = row[self.features].to_numpy(dtype=np.float32)
        res = torch.tensor(x, dtype=float32)
        return res, torch.tensor(row['isFraud'], dtype=torch.float32).flatten()

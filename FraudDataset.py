import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
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
        self.df = pd.read_csv(path, dtype=np.float32, header=0)
        scalar = MinMaxScaler()
        self.df['nameOrig'].apply(parseName)
        self.df['nameDest'].apply(parseName)
        self.df['type'].apply(parseType)
        self.features = [
            'step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest',
            'newbalanceDest',
        ]
        self.df[self.features] = scalar.fit_transform(self.df[self.features])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        res = torch.tensor(self.df[self.features])
        return res, self.df['isFraud']

from torch.utils.data import Dataset
import torch.nn as nn
import torch
import pandas as pd

# dataset
class PokerDataset(Dataset):

    def __init__(self):
        df = pd.read_csv('./dataset.csv')
        features = [self.card_to_features(c) for c in df['card1']] + \
                   [self.card_to_features(c) for c in df['card2']]
        # each row: [rank1, suit1, rank2, suit2]
        X = [[*self.card_to_features(c1), *self.card_to_features(c2)]
             for c1, c2 in zip(df['card1'], df['card2'])]
        self.x = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(df['win_rate'].values, dtype=torch.float32).unsqueeze(1)
        self.n_samples = len(df)

    def card_to_features(self, card):
        ranks = {'2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 'T':10, 'J':11, 'Q':12, 'K':13, 'A':14}
        suits = {'h': 0, 'd': 1, 'c': 2, 's': 3}
        return [ranks[card[0]], suits[card[1]]]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

# model
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model,self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
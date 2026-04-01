import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from model import PokerDataset, Model

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/poker")

from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f"runs/poker_{timestamp}")

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = PokerDataset()
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=0)

# hyperparameters
hidden_layer = 20
learning_rate = 0.01
num_epochs = 50

model = Model(4, hidden_layer, 1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),learning_rate)


# training loop
for epoch in range(num_epochs):
    for i, (hands, labels) in enumerate(train_loader):
        hands,labels = hands.to(device), labels.to(device)
        outputs = model(hands)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)

# test

model.eval()
with torch.no_grad():
    n_cumulative_div = 0
    n_samples = 0
    for hands, labels in test_loader:
        hands,labels = hands.to(device), labels.to(device)
        outputs     = model(hands)
        n_samples += 1
        n_cumulative_div += criterion(outputs,labels).item()

    print(f'loss rate: {n_cumulative_div/n_samples:.4f}')

writer.flush()
writer.close()

torch.save(model.state_dict(), 'poker_model.pth')
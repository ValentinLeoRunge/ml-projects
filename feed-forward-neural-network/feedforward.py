import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
input_size = 784
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset  = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1   = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2   = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


model     = NeuralNet(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ── track every single step ──────────────────────────────────────────────────
loss_history  = []
acc_history   = []
step_history  = []
global_step   = 0
n_total_steps = len(train_loader)
epoch_boundaries = []   # step numbers where a new epoch starts

# ── training loop ─────────────────────────────────────────────────────────────
for epoch in range(num_epochs):
    epoch_boundaries.append(global_step)
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss    = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step += 1

        # record EVERY step for a smooth graph
        _, preds    = torch.max(outputs, 1)
        batch_acc   = 100.0 * (preds == labels).sum().item() / labels.size(0)
        loss_history.append(loss.item())
        acc_history.append(batch_acc)
        step_history.append(global_step)

        if (i + 1) % 100 == 0:
            print(f"epoch {epoch+1}/{num_epochs}, "
                  f"step {i+1}/{n_total_steps}, "
                  f"loss = {loss.item():.4f}, "
                  f"batch acc = {batch_acc:.1f}%")

# final test accuracy
# collect ALL test images + labels for random sampling later
all_test_images  = []
all_test_labels  = []
all_test_preds   = []
all_test_outputs = []

model.eval()
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images_flat = images.reshape(-1, 28 * 28).to(device)
        labels      = labels.to(device)
        outputs     = model(images_flat)
        _, preds    = torch.max(outputs, 1)

        n_samples += labels.shape[0]
        n_correct += (preds == labels).sum().item()

        all_test_images.append(images.cpu())
        all_test_labels.append(labels.cpu())
        all_test_preds.append(preds.cpu())
        all_test_outputs.append(outputs.cpu())

final_acc = 100.0 * n_correct / n_samples
print(f'\nFinal test accuracy = {final_acc:.2f}%')

# flatten all test data into single tensors
all_test_images  = torch.cat(all_test_images,  dim=0)   # [10000, 1, 28, 28]
all_test_labels  = torch.cat(all_test_labels,  dim=0)   # [10000]
all_test_preds   = torch.cat(all_test_preds,   dim=0)   # [10000]
all_test_outputs = torch.cat(all_test_outputs, dim=0)   # [10000, 10]

# visualising learning curve graph

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Learning Curve', fontsize=16, fontweight='bold')

ax1.plot(step_history, loss_history, color='#e74c3c', linewidth=1.2, alpha=0.8)
ax1.set_title('Loss pro Step')
ax1.set_xlabel('Step')
ax1.set_ylabel('Cross Entropy Loss')
ax1.grid(True, alpha=0.3)
for b in epoch_boundaries[1:]:
    ax1.axvline(x=b, color='gray', linestyle='--', alpha=0.6, label='New Epoch')
handles, lbls = ax1.get_legend_handles_labels()
if handles:
    ax1.legend(handles[:1], lbls[:1])

ax2.plot(step_history, acc_history, color='#2ecc71', linewidth=1.2, alpha=0.8)
ax2.set_title('Batch Accuracy per Step')
ax2.set_xlabel('Step')
ax2.set_ylabel('Accuracy (%)')
ax2.set_ylim(0, 100)
ax2.axhline(y=final_acc, color='#3498db', linestyle='--', linewidth=1.8,
            label=f'Final Test Acc: {final_acc:.1f}%')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('learning_curve.png', dpi=150, bbox_inches='tight')
plt.show()

# visualizing random samples and decisions

indices = random.sample(range(len(all_test_images)), 20)

fig2, axes = plt.subplots(4, 5, figsize=(14, 11))
fig2.suptitle(f'AI classifying 20 random inputs  |  Overall accuracy: {final_acc:.1f}%',
              fontsize=14, fontweight='bold')

for plot_idx, data_idx in enumerate(indices):
    ax         = axes.flat[plot_idx]
    img        = all_test_images[data_idx][0].numpy()
    true_label = all_test_labels[data_idx].item()
    pred_label = all_test_preds[data_idx].item()
    correct    = pred_label == true_label

    ax.imshow(img, cmap='gray')
    ax.axis('off')

    color  = '#27ae60' if correct else '#e74c3c'
    symbol = '✓' if correct else '✗'
    ax.set_title(f'AI: {pred_label} {symbol}\nTrue: {true_label}',
                 fontsize=10, color=color, fontweight='bold')

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(color)
        spine.set_linewidth(3)

plt.tight_layout()
plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
plt.show()

# visualising estimated probability distribution for random images
indices6 = random.sample(range(len(all_test_images)), 6)

fig3, axes3 = plt.subplots(2, 3, figsize=(14, 8))
fig3.suptitle('Estimated Probability Distribution for Random Images', fontsize=14, fontweight='bold')

for plot_idx, data_idx in enumerate(indices6):
    ax = axes3.flat[plot_idx]
    img = all_test_images[data_idx][0].numpy()
    true_label = all_test_labels[data_idx].item()
    pred_label = all_test_preds[data_idx].item()
    logits = all_test_outputs[data_idx]
    probs = F.softmax(logits, dim=0).numpy()
    correct = pred_label == true_label

    # input image — left, vertically centered next to y axis
    ax_img = ax.inset_axes([-0.07, 0.25, 0.25, 0.45])
    ax_img.imshow(img, cmap='gray')
    ax_img.axis('off')

    # bar chart — smaller, shifted right to give image space
    ax_bar = ax.inset_axes([0.32, 0.08, 0.60, 0.75])
    bar_colors = ['#e74c3c' if i == pred_label else '#bdc3c7' for i in range(10)]
    ax_bar.bar(range(10), probs, color=bar_colors)
    ax_bar.set_xticks(range(10))
    ax_bar.set_xlabel('Class')
    ax_bar.set_ylabel('Probability')
    ax_bar.set_ylim(0, 1)
    ax_bar.grid(True, alpha=0.2, axis='y')

    color = '#27ae60' if correct else '#e74c3c'
    symbol = '✓' if correct else '✗'
    ax.set_title(f'Predicted: {pred_label} {symbol}  |  True: {true_label}',
                 color=color, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig('confidence.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nGenerated images for training performance analysis:")
print("  learning_curve.png")
print("  predictions.png")
print("  confidence.png")
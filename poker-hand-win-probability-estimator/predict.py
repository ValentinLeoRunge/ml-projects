import torch
from model import Model, PokerDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Model(4, 20, 1)
model.load_state_dict(torch.load('poker_model.pth', weights_only=True))
model.to(device)
model.eval()

def predict(card1, card2):
    dataset = PokerDataset()
    features = dataset.card_to_features(card1) + dataset.card_to_features(card2)
    tensor = torch.tensor(features, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(tensor)
    print(f'win probability with {card1} and {card2}: {output.item():.2f}')

print("Poker Win Probability Estimator -- AI")
print("Tell me your card and I will estimate your win probability")
print("Format: <card1> <card2> (e.g. 'Ah Ks')")
print("Ranks: 2-9, T, J, Q, K, A | Suits: h, d, c, s")
print("Type 'exit' to quit\n")

def is_valid_card(card):
    ranks = {'2','3','4','5','6','7','8','9','T','J','Q','K','A'}
    suits = {'h','d','c','s'}
    return len(card) == 2 and card[0] in ranks and card[1] in suits

while True:
    user_input = input("Enter hand: ")
    if user_input.lower() == 'exit':
        break
    parts = user_input.strip().split()
    if len(parts) != 2:
        print("please enter exactly 2 cards")
        continue
    if not is_valid_card(parts[0]) or not is_valid_card(parts[1]):
        print("invalid card — use format like 'Ah Ks' (rank + suit)")
        continue
    if parts[0] == parts[1]:
        print("disclaimer: both cards cannot be the same normally but I will still estimate the fictive win probability for you")
    predict(parts[0], parts[1])
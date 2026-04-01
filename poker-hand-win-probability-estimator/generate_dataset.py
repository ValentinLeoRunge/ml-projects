import random
from treys import Card, Evaluator
import pandas as pd

# signs and numbers for card generation
signs = ['h','d','c','s']
numbers = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']

# deal random cards
def deal_cards(nr):
    if nr < 0 or nr > 53:
        nr = 2

    cards = []

    while nr > len(cards):
        # generate a random card
        random_sign = random.randint(0,3)
        random_number = random.randint(0,12)
        card = numbers[random_number] + signs[random_sign]

        # add card to deck
        if card not in cards:   # this gets very inefficient if you try to draw a lot of cards but is efficient enough for drawing just a few cards
            cards.append(card)

    return cards

# deal 5 cards for the opponent and the community
def deal_others_cards(hand):
    nr = 7
    cards = []

    while nr > len(cards):
        # generate a random card
        random_sign = random.randint(0,3)
        random_number = random.randint(0,12)
        card = numbers[random_number] + signs[random_sign]

        # add card to deck
        if card not in cards and card not in hand:  # this gets very inefficient if you try to draw a lot of cards but is efficient enough for drawing just a few cards
            cards.append(card)

    return cards


def simulate_rounds(str_hand,nr):
    evaluator = Evaluator()
    win_cnt = 0

    hand = [Card.new(c) for c in str_hand]
    for i in range(nr):
        str_cards = deal_others_cards(str_hand)   # first two indices represent the opponents hand, last five the community cards
        cards = [Card.new(c) for c in str_cards]
        op_hand = cards[0:2]
        com_cards = cards[2:7]

        score = evaluator.evaluate(com_cards, hand)
        op_score = evaluator.evaluate(com_cards, op_hand)

        if score < op_score:    # placeholder for comparison
            win_cnt += 1

    return win_cnt / nr

def create_csv(nr_datapoints):
    rows = []
    for i in range(nr_datapoints):
        hand = deal_cards(2)
        win_rate = simulate_rounds(hand,1000)
        rows.append({'card1': hand[0], 'card2': hand[1], 'win_rate': win_rate})

    df = pd.DataFrame(rows)
    df.to_csv('dataset.csv', index=False)

#cards = deal_cards(7)
#for c in cards:
#    print(f'{c}')

#score = simulate_rounds(['Ah', 'As'],10)
#print(f'score: {score}')

create_csv(10000)
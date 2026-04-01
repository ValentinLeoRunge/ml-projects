import random

# signs and numbers for card generation
signs = ['h','d','c','s']
numbers = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']

# deal 2-5 cards
def deal_cards(nr):
    if nr < 0 or nr > 9:
        nr = 9

    cards = []

    while nr > len(cards):
        # generate a random card
        random_sign = random.randint(0,3)
        random_number = random.randint(0,12)
        card = [numbers[random_number], signs[random_sign]]

        # add card to deck
        if card not in cards:
            cards.append(card)

    return cards


def simulate_rounds(nr):
    win_cnt = 0
    for i in range(nr):
        cards = deal_cards(9)   # first two indices represent own hand, second two the opponents hand, last five the community cards
        hand = cards[0:2]
        op_hand = cards[2:4]
        com_cards = cards[4:9]

        if True:    # placeholder for comparison
            win_cnt += 1

    return win_cnt / nr
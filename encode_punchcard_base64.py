"""
Simple Base64 Punch Card Encoder
Maps each base64 character to a unique punch pattern
"""

import base64
import json

# Read code
with open('attogpt.py', 'r') as f:
    code = f.read()

# Convert to base64
code_bytes = code.encode('utf-8')
b64 = base64.b64encode(code_bytes).decode('ascii')

print(f"Original: {len(code)} chars")
print(f"Base64: {len(b64)} chars")
print(f"Expansion: {len(b64)/len(code):.2f}x")

# Base64 alphabet: A-Z, a-z, 0-9, +, /, =
# Map each to a unique punch pattern (we have 12 rows, so 2^12 = 4096 possible patterns)

def char_to_punch_simple(char):
    """Simple mapping: use ASCII value directly"""
    ascii_val = ord(char)
    punches = []
    for bit in range(12):
        if ascii_val & (1 << bit):
            punches.append(bit)
    return punches

# Encode
COLS_PER_CARD = 80
cards = []
current_card = []

for char in b64:
    if len(current_card) >= COLS_PER_CARD:
        cards.append(current_card)
        current_card = []
    current_card.append(char_to_punch_simple(char))

if current_card:
    while len(current_card) < COLS_PER_CARD:
        current_card.append([])
    cards.append(current_card)

print(f"Cards needed: {len(cards)}")

# Save
with open('punchcard_encoding.json', 'w') as f:
    json.dump({
        'encoding': 'base64',
        'num_cards': len(cards),
        'chars_per_card': COLS_PER_CARD,
        'total_chars': len(b64),
        'original_size': len(code),
        'base64_data': b64,
        'cards': [[sorted(col) for col in card] for card in cards]
    }, f)

print("Encoding saved!")
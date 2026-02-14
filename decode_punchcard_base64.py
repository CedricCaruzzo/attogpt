#!/usr/bin/env python3
"""
Base64 Punch Card Decoder
"""

import json
import base64

print("=" * 60)
print("ATTOGPT PUNCH CARD DECODER (Base64)")
print("=" * 60)

with open('punchcard_encoding.json', 'r') as f:
    data = json.load(f)

print(f"\nEncoding: {data['encoding']}")
print(f"Cards: {data['num_cards']}")
print(f"Original size: {data['original_size']} chars")

def punches_to_char(punches):
    """Convert punch pattern back to ASCII character"""
    ascii_val = 0
    for bit in punches:
        ascii_val |= (1 << bit)
    return chr(ascii_val)

# Decode
decoded_b64 = []
for card in data['cards']:
    for col in card:
        if col:  # Skip empty columns
            decoded_b64.append(punches_to_char(col))

decoded_b64_str = ''.join(decoded_b64)

# Verify against stored base64
if decoded_b64_str == data['base64_data']:
    print("Base64 decoded successfully from punch cards!")
else:
    print("Base64 mismatch")

# Decode base64 to original
decoded_bytes = base64.b64decode(decoded_b64_str)
decoded_text = decoded_bytes.decode('utf-8')

print(f"Decoded: {len(decoded_text)} characters")

# Save
with open('attogpt_decoded.py', 'w') as f:
    f.write(decoded_text)

print("Saved to: attogpt_decoded.py")
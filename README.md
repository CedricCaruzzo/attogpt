# Attogpt

**A complete GPT neural network in 31 lines of Python, encoded in vintage IBM punch cards.**

From [Andrej Karpathy's microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) (240 lines) → **attogpt** (31 lines) — an **87.1% reduction** while maintaining full functionality.

## The Challenge

How small can a working GPT get?

- **microgpt** by Andrej Karpathy: 240 lines
- **picogpt**: 64 lines (shared as QR code)
- **femtogpt**: 53 lines (shared as 3000-digit prime number)
- **attogpt**: **31 lines** (shared as 52 IBM punch cards!)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/attogpt.git
cd attogpt

# Run the model (trains on names dataset)
python3 attogpt.py
```

### What It Does

After 1000 training steps, attogpt generates plausible new names:

```
step  996 / 1000 | loss 2.1018
step  997 / 1000 | loss 1.7791
step  998 / 1000 | loss 2.4764
step  999 / 1000 | loss 2.4730
step 1000 / 1000 | loss 2.6497

--- inference ---
sample  1: kamon
sample  2: ann
sample  3: karai
sample  4: jaire
sample  5: vialan
sample  6: karia
sample  7: yeran
sample  8: anna
sample  9: areli
sample 10: kaina
```

## What's Included (Despite Being Only 31 Lines)

- **Autograd Engine**: Full backpropagation with computational graph
- **Transformer Architecture**: Multi-head self-attention mechanism
- **Training Loop**: Adam optimizer with bias correction
- **Text Generation**: Temperature-controlled sampling
- **Position Embeddings**: Learned positional encoding
- **RMS Normalization**: Layer normalization for stability
- **Feed-Forward Networks**: MLP blocks with ReLU activation

All in **3,117 characters** using only Python's standard library.

## The Punch Card Encoding

To showcase the compression, attogpt is encoded in **52 IBM punch cards** — the same format that powered computing in the 1960s and helped send humans to the moon. (Well, revisited due to lower cases incompatibility...)

### How It Works

1. **Python code** → UTF-8 bytes
2. **UTF-8 bytes** → Base64 encoding
3. **Base64 characters** → 12-bit punch patterns
4. **80 columns per card** × 52 cards = complete neural network

### Try It Yourself

```bash
# View the encoding
cat punchcard_encoding.json

# Decode back to Python
python3 decode_punchcard_base64.py

```

## 📁 Repository Structure

```
attogpt/
├── attogpt.py                      # The 31-line GPT implementation
├── microgpt.py                     # Original 240-line version (for comparison)
├── picotgpt.py                     # 68-line version
├── femtogpt.py                     # 53-line version
├── punchcard_encoding.json         # All 52 punch card patterns
├── encode_punchcard_base64.py      # Script to encode Python → punch cards
├── decode_punchcard_base64.py      # Script to decode punch cards → Python
├── input.txt                       # Names dataset (auto-downloaded if missing)
└── README.md                       # You are here
```

## 🔧 Technical Details

### Model Architecture

```
Embedding:     16-dimensional
Attention:     4 heads, 1 layer
Context:       16 tokens
Parameters:    ~5,000 total
Block size:    16 tokens
Vocabulary:    28 tokens (26 letters + BOS + space)
```

### Compression Techniques

- **Lambda functions**: Replace method definitions
- **Dictionary state**: Flatten parameter structure  
- **Expression folding**: Combine operations into single statements
- **Variable reuse**: Aggressive name reuse
- **Inline operations**: Eliminate intermediate variables

Despite the compression, attogpt produces **identical results** to microgpt.

## Comparison

| Version | Lines | Reduction | Shared As |
|---------|-------|-----------|-----------|
| microgpt | 240 | baseline | Text file |
| picogpt | 64 | -73.4% | QR code |
| femtogpt | 53 | -77.9% | Prime number |
| **attogpt** | **31** | **-87.1%** | **52 punch cards** |

## What You Can Learn

This project demonstrates:

1. **How small can neural networks get?** Surprisingly small while staying functional
2. **Code compression techniques** for Python
3. **Historical data formats** and how far storage has evolved
4. **The elegance of transformers** distilled to their essence
5. **Reversible encoding schemes** for data preservation

## Dependencies

**None!** attogpt uses only Python's standard library:
- `os` - File operations
- `math` - Mathematical functions  
- `random` - Random number generation
- `urllib.request` - Dataset download (automatic)

Tested on Python 3.8+

## How to Experiment

### Modify the training

```python
# In attogpt.py, adjust these parameters:
num_steps = 1000      # Number of training steps
learning_rate = 0.01  # Learning rate
temperature = 0.5     # Generation temperature (higher = more creative)
```

### Train on your own data

Replace `input.txt` with your own text file (one entry per line):

```python
# attogpt will automatically use your custom dataset
docs = [l.strip() for l in open('input.txt').read().strip().split('\n')]
```

### Compare to the originals

Run all versions to see the outputs:

```bash
python3 microgpt.py   # 240 lines, original
python3 picotgpt.py   # 64 lines  
python3 femtogpt.py   # 53 lines
python3 attogpt.py    # 31 lines (ours!)
```

## 🤝 Acknowledgments

- **Andrej Karpathy** - Original [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) (240 lines)
- **Kuber** - [picogpt](https://github.com/Kuberwastaken/picogpt) (64 lines, QR code format)
- **MatchaOnMuffins** - [femtogpt](https://github.com/MatchaOnMuffins/femtoGPT) (53 lines, prime number format)

This project builds on their excellent work to push compression even further.
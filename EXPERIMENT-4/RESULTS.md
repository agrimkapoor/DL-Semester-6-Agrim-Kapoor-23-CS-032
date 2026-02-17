# Model Comparison Report

## Models Implemented

| # | Model |
|---|-------|
| 1 | RNN + One-Hot Encoding |
| 2 | LSTM + One-Hot Encoding |
| 3 | RNN + Trainable Embedding |
| 4 | LSTM + Trainable Embedding |

---

## Training Results

| Model | Final Loss | Training Time |
|---|---|---|
| RNN + One-Hot | 1.1056 | 390.12 sec |
| LSTM + One-Hot | 0.7431 | 1404.47 sec |
| RNN + Embedding | 0.8420 | 295.18 sec |
| **LSTM + Embedding** | **0.6679** | **464.73 sec** |

---

## Generated Text Samples

**Prompt:** `"love is like"`

### 1️. RNN + One-Hot

> love is like the thought of the atlantic are of me it was again you the moonlight man the fibre of shapes planets in the evening elements of my light place to me yet there is a name or yellow a moment unsaid

### 2️. LSTM + One-Hot

> love is like me and was what to make the far of water and and hollow what is make my than head the head has flow on and free had is one to one at but work it shall you me the eyes

### 3️. RNN + Embedding

> love is like a thousand christmas trees at last we're tired my heart and i we dealt with books we trusted men and in the night of the belly the angels with her sewing machine or in the factory or mill the paving

### 4️. LSTM + Embedding

> love is like water the air or the atlantic or like poetry it was many and many a year ago in a kingdom by the sea that the wind came out of the cloud by night chilling and killing my annabel lee but

---

## Observations

- **One-Hot Encoding** produced higher loss compared to Embedding models because one-hot vectors are high-dimensional and sparse.
- **Embedding models** trained faster and converged better since embeddings learn meaningful dense representations.
- **RNN** struggled with long-term dependencies and often repeated words.
- **LSTM** performed better than RNN because it maintains memory through gating mechanisms.
- **LSTM with Embedding** achieved the lowest loss and produced the most coherent and meaningful text.

---

## Conclusion

Among all models, **LSTM with Trainable Embeddings** performed the best in terms of:

| Metric | Result |
|---|---|
| Training Loss | Lowest (0.6679) |
| Convergence | Fastest & most stable |
| Text Quality | Most coherent and meaningful |

This shows that combining memory-based sequence modeling (LSTM) with dense word representations (Embeddings) is most effective for text generation tasks.
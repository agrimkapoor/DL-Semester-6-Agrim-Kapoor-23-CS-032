import torch

def generate(model, start_text, word_to_idx, idx_to_word, seq_length, length=20):
    model.eval()
    words = start_text.lower().split()

    for _ in range(length):
        seq = [word_to_idx.get(w, 0) for w in words[-seq_length:]]
        seq_tensor = torch.tensor([seq])

        with torch.no_grad():
            output = model(seq_tensor)
            predicted = torch.argmax(output, dim=1).item()

        words.append(idx_to_word[predicted])

    return " ".join(words)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

from config import *
from utils.preprocessing import load_text_from_csv, build_vocab, create_sequences
from utils.dataset import TextDataset
from models.rnn_onehot import RNNOneHot # change model here
from utils.generate import generate


def train():

    # Load and preprocess data
    words = load_text_from_csv("data/poem.csv")
    vocab, word_to_idx, idx_to_word = build_vocab(words)
    X, y = create_sequences(words, word_to_idx, SEQUENCE_LENGTH)

    dataset = TextDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model
    model = RNNOneHot(len(vocab), HIDDEN_SIZE) # change here


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Measure training time
    start_time = time.time()

    for epoch in range(EPOCHS):
        total_loss = 0

        for batch_X, batch_y in loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    end_time = time.time()
    training_time = end_time - start_time

    print(f"\nTraining Time: {training_time:.2f} seconds")

    # Save model
    torch.save(model.state_dict(), "model.pth")

    # Generate text
    print("\nGenerated Text Sample:\n")

    sample = generate(
        model,
        "love is like",
        word_to_idx,
        idx_to_word,
        SEQUENCE_LENGTH,
        length=40
    )

    print(sample)


if __name__ == "__main__":
    train()

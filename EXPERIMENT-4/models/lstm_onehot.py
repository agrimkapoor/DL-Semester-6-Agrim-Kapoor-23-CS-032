import torch.nn as nn
import torch.nn.functional as F

class LSTMOneHot(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(vocab_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, x):
        x = F.one_hot(x, num_classes=self.vocab_size).float()
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

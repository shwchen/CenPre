import torch.nn as nn
import torch.nn.functional as f


class SimpleClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(SimpleClassifier, self).__init__()

        self.linear1 = nn.Linear(embedding_dim, 512)
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(p=0.1)

        self.linear2 = nn.Linear(512, 256)
        self.batch_norm2 = nn.BatchNorm1d(256)

        self.linear3 = nn.Linear(256, num_classes)

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, embeddings):
        x = self.leaky_relu(self.batch_norm1(self.linear1(embeddings)))
        x = self.dropout(x)
        x = self.leaky_relu(self.batch_norm2(self.linear2(x)))
        logits = self.linear3(x)
        return logits


class Discriminator(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Discriminator, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = f.relu(self.lin1(x))
        x = f.relu(self.lin2(x))
        x = self.lin3(x)
        return x

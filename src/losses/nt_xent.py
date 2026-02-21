import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        batch_size = z1.size(0)

        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Concatenate embeddings
        z = torch.cat([z1, z2], dim=0)  # shape: [2N, dim]

        # Similarity matrix
        sim_matrix = torch.matmul(z, z.T)  # cosine similarity since normalized

        # Remove self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)

        # Positive pairs
        positives = torch.cat([
            torch.sum(z1 * z2, dim=1),
            torch.sum(z2 * z1, dim=1)
        ], dim=0)

        # Divide by temperature
        logits = sim_matrix / self.temperature
        positives = positives / self.temperature

        # Labels: positive pair is always index 0 after masking trick
        labels = torch.arange(2 * batch_size).to(z.device)
        labels = (labels + batch_size) % (2 * batch_size)

        loss = F.cross_entropy(logits, labels)

        return loss

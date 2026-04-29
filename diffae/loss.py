import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07, aggregation_fn=torch.sum):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.aggregation_fn = aggregation_fn

    def _normalized_similarity_matrix(self, features: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, dim=-1)
        similarity_matrix = torch.matmul(features, features.T)
        return similarity_matrix

    def _calculate_info_nce_loss(
        self, positives: torch.Tensor, negatives: torch.Tensor
    ) -> torch.Tensor:
        positives = positives / self.temperature
        negatives = negatives / self.temperature
        logits = torch.cat([positives, negatives], dim=1)

        numerator = self.aggregation_fn(torch.exp(positives), dim=-1)
        denominator = self.aggregation_fn(torch.exp(logits), dim=-1)

        loss = -torch.log(numerator / (denominator + 1e-8) + 1e-8)

        return loss

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:

        similarity_matrix = self._normalized_similarity_matrix(features)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool, device=features.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(similarity_matrix.shape[0], -1)

        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        # scale by temperature
        loss = self._calculate_info_nce_loss(positives, negatives)

        return loss


class MaskedNegativesInfoNCELoss(InfoNCELoss):
    def __init__(self, temperature: float = 0.07, aggregation_fn=torch.sum):
        super(MaskedNegativesInfoNCELoss, self).__init__(temperature, aggregation_fn)

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        negative_mask: torch.Tensor,
    ) -> torch.Tensor:

        similarity_matrix = self._normalized_similarity_matrix(features)

        self_similarity_mask = torch.eye(
            labels.shape[0],
            dtype=torch.bool,
            device=features.device,
        )

        labels = labels[~self_similarity_mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~self_similarity_mask].view(
            similarity_matrix.shape[0], -1
        )

        positives = similarity_matrix[labels].view(similarity_matrix.shape[0], -1)

        negative_mask = negative_mask[~self_similarity_mask].view(  # type:ignore
            negative_mask.shape[0], -1
        )

        # mask out negative views coming from other subjects
        negative_labels = ~labels & negative_mask

        negatives = similarity_matrix[negative_labels].view(negative_labels.shape[0], -1)

        # scale by temperature
        loss = self._calculate_info_nce_loss(positives, negatives)

        return loss

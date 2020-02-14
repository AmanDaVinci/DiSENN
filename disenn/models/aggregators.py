import torch
import torch.nn as nn
import torch.nn.functional as F


class SumAggregator(nn.Module):
    def __init__(self, num_classes):
        """Basic Sum Aggregator that aggregates the concepts and relevances by summing their products.
        """
        super().__init__()
        self.num_classes = num_classes

    def forward(self, concepts, relevances):
        """Forward pass of Sum Aggregator

        Aggregates the sum product of concepts and relevances to return the predictions for each class.

        Parameters
        ----------
        concepts : torch.Tensor
            concepts generated by the Conceptizer of shape (batch_size, num_concepts)
        
        relevances : torch.Tensor
            relevance parameters generated by the parameterizer of shape (batch_size, num_concepts, num_classes)

        Returns
        -------
        class_predictions : torch.Tensor
            Predictions for each class of shape (batch_size, num_classes)
            
        """
        aggregates = torch.bmm(relevances.permute(0, 2, 1), concepts)
        return F.log_softmax(aggregates, dim=1)
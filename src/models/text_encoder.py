"""Text encoder for customer communication data."""

import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    """Encodes text data into a fixed-size representation.

    Uses a pretrained Transformer model (e.g., BERT) to extract
    semantic features from support tickets, feedback, and chat logs.

    Attributes:
        pretrained_model: Name of the pretrained Transformer model.
        hidden_size: Dimension of the output representation.
        freeze_layers: Number of Transformer layers to freeze.
    """

    def __init__(
        self,
        pretrained_model: str = "bert-base-uncased",
        hidden_size: int = 768,
        freeze_layers: int = 10,
    ) -> None:
        """Initialize the text encoder.

        Args:
            pretrained_model: HuggingFace model identifier.
            hidden_size: Dimension of the encoder output.
            freeze_layers: Number of bottom layers to freeze during training.
        """
        super().__init__()
        self.transformer = AutoModel.from_pretrained(pretrained_model)
        self.hidden_size = hidden_size

        for i, layer in enumerate(self.transformer.encoder.layer):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode text input.

        Args:
            input_ids: Token IDs of shape (batch_size, max_length).
            attention_mask: Attention mask of shape (batch_size, max_length).

        Returns:
            Encoded representation of shape (batch_size, hidden_size).
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding

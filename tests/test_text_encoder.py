"""Unit tests for TextEncoder."""

import pytest
import torch


@pytest.fixture(scope="module")
def text_encoder():
    """Create a TextEncoder instance, skip if import fails."""
    try:
        from src.models.text_encoder import TextEncoder
        encoder = TextEncoder(
            pretrained_model="prajjwal1/bert-tiny",
            hidden_size=128,
            freeze_layers=0,
        )
        return encoder
    except Exception as e:
        pytest.skip(f"Could not load TextEncoder: {e}")


class TestTextEncoder:
    """Tests for TextEncoder class."""

    def test_output_shape(self, text_encoder) -> None:
        """Test that forward returns correct output shape."""
        batch_size = 4
        seq_len = 64
        hidden_size = 128

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        output = text_encoder(input_ids, attention_mask)

        assert output.shape == (batch_size, hidden_size)

    def test_with_padding(self, text_encoder) -> None:
        """Test encoder handles padded sequences correctly."""
        batch_size = 4
        seq_len = 64
        hidden_size = 128

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        attention_mask[:, 32:] = 0

        output = text_encoder(input_ids, attention_mask)

        assert output.shape == (batch_size, hidden_size)

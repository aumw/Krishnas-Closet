# config.py

class VisionConfig:
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=(196, 392),
        patch_size=14,
        layer_norm_eps=1e-6,
        attention_dropout=0.2,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout

class TextDecoderConfig:
    def __init__(
        self,
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=6,
        vocab_size=30522,  # Update with your vocabulary size
        layer_norm_eps=1e-6,
        dropout=0.2,
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.dropout = dropout

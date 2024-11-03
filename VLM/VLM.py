# vision_language_model.py
import torch
import torch.nn as nn
from config import VisionConfig, TextDecoderConfig
from vision_encoder import SiglipVisionModel

class VisionLanguageModel(nn.Module):
    def __init__(self, vision_config: VisionConfig, text_decoder_config: TextDecoderConfig):
        super().__init__()
        self.vision_encoder = SiglipVisionModel(vision_config)  # Use VisionConfig
        self.text_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=text_decoder_config.hidden_size,
                nhead=text_decoder_config.num_attention_heads,
                dropout=text_decoder_config.dropout,
            ),
            num_layers=text_decoder_config.num_hidden_layers,
        )
        self.token_embedding = nn.Embedding(text_decoder_config.vocab_size, text_decoder_config.hidden_size)
        self.fc_out = nn.Linear(text_decoder_config.hidden_size, text_decoder_config.vocab_size)

    def forward(self, pixel_values, input_ids):
        vision_features = self.vision_encoder(pixel_values)
        embedded_input_ids = self.token_embedding(input_ids)


                # Add print statements to debug shapes
        #print(f"vision_features shape: {vision_features.shape}")
        #print(f"embedded_input_ids shape: {embedded_input_ids.shape}")
        
        output = self.text_decoder(embedded_input_ids, vision_features)
        output = self.fc_out(output)
        return output

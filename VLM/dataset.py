# dataset.py
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torchvision import transforms  # Import transforms from torchvision
from PIL import Image
import os
import json

class VisionLanguageDataset(Dataset):
    def __init__(self, images_dir, annotations_file):
        self.images_dir = images_dir
        self.annotations = self.load_annotations(annotations_file)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Update as needed
        self.preprocess = transforms.Compose([
            transforms.Resize((196, 392)),      # Resize to match model input
            transforms.ToTensor(),              # Convert to tensor
            transforms.Normalize(               # Normalize based on ImageNet standards
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def load_annotations(self, annotations_file):
        # Load annotations as a list of {image_filename, caption}
        # This will depend on your data format
        with open(annotations_file) as jsonfile:
            annotations = json.load(jsonfile)
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations[idx]
        
        # Load and preprocess image
        image = Image.open(os.path.join(self.images_dir, item['image_filename'])).convert("RGB")
        pixel_values = self.preprocess_image(image)
        
        # Tokenize caption and get input_ids
        caption = item['caption']
        input_ids = self.tokenizer(caption, return_tensors="pt", padding="max_length", max_length=392).input_ids.squeeze(0)
        
        # Use input_ids as labels (if you don't have separate labels)
        labels = input_ids.clone()  # or define separate labels if applicable
        labels[:-1] = input_ids[1:]  # Shift tokens to the right
        labels[-1] = -100  # Padding token to ignore the last prediction
        
        return pixel_values, input_ids, labels

    def preprocess_image(self, image):
        # Preprocess image to match model requirements (resize, normalize)
        return self.preprocess(image)

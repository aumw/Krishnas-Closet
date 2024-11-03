import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from config import VisionConfig, TextDecoderConfig
from VLM import VisionLanguageModel
from dataset import VisionLanguageDataset

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_training(rank, world_size, epochs=15, batch_size=16, learning_rate=1e-4):
    setup(rank, world_size)

    # Initialize configuration
    vision_config = VisionConfig()
    text_decoder_config = TextDecoderConfig()

    # Dataset and Dataloader setup
    dataset = VisionLanguageDataset(images_dir='', annotations_file='dataset.json')
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    print(dataloader)

    # Model and optimizer setup
    model = VisionLanguageModel(vision_config, text_decoder_config).to(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()

    # Training Loop
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # Ensures data shuffling is different across epochs
        epoch_loss = 0.0  # To accumulate loss
        batch_count = 0   # To keep track of number of batches
        
        for batch in dataloader:
            print(len(batch))
            pixel_values = batch[0].to(rank)  # Access first element (pixel_values)
            input_ids = batch[1].to(rank)     # Access second element (input_ids)
            labels = batch[2].to(rank)        # Access third element (labels)
    
            # Forward pass
            outputs = model(pixel_values, input_ids)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
    
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # Accumulate loss and batch count
            epoch_loss += loss.item()
            batch_count += 1
    
        # Calculate average loss for the epoch
        avg_epoch_loss = epoch_loss / batch_count
    
        # Print average loss at the end of each epoch
        if rank == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_epoch_loss:.4f}")
    
        # Save model after every epoch (only on rank 0 to avoid multiple saves)
        if rank == 0:
            model_save_path = f"vision_language_model_full_v4_{epoch+1}.pth"
            torch.save(model.module, model_save_path)
            print(f"Model saved at {model_save_path}")
    
    # Final cleanup
    cleanup()

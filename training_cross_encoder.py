import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import json
import os
from cross_encoder_hf_cls import PoliticalBiasCrossEncoder
from tqdm import tqdm
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

class PoliticalBiasDataset(Dataset):
    def __init__(self, texts, indicators, labels):
        """
        Args:
            texts: List of article texts
            indicators: List of indicator texts
            labels: List of labels
        """
        self.texts = texts
        self.indicators = indicators
        self.labels = labels
        assert len(texts) == len(indicators) == len(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indicator = self.indicators[idx]
        label = self.labels[idx]
        
        return {
            "text": text,
            "indicator": indicator,
            "label": label
        }

def train_cross_encoder(
    model,
    train_loader,
    val_loader,
    num_epochs=10,
    learning_rate=2e-5,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_dir="checkpoints",
):
    """
    Train the cross encoder model with wandb monitoring
    """
    # Initialize wandb
    wandb.init(
        project="political-bias-cross-encoder-new-bignews-16",
        config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "model_name": model.model_name,
            "batch_size": train_loader.batch_size,
            "device": device
        }
    )
    
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_loss = float('inf')
    global_step = 0
    save_interval = 25000  # Save every 25k steps
    max_eval_samples = 30000  # Maximum number of validation samples to evaluate
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        train_steps = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in progress_bar:
            texts = batch["text"]
            indicators = batch["indicator"]
            labels = batch["label"].to(device).float()
            
            # Assert all labels are >= 0
            assert torch.all(labels >= 0), "Found negative labels in batch!"
            
            optimizer.zero_grad()
            outputs = model(texts, indicators)
            outputs = outputs.squeeze(1)
            
            # Add prediction printing every 1000 steps
            if train_steps % 1000 == 0:
                print("\nSample predictions:")
                for i in range(min(3, len(outputs))):
                    print(f"Prediction: {outputs[i].item():.3f}, Label: {labels[i].item():.3f}")
                print()
            
            assert outputs.shape == labels.shape
            loss = model.compute_mse_loss(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_steps += 1
            global_step += 1
            
            # Log batch loss to wandb
            wandb.log({"batch_loss": loss.item(), "global_step": global_step})
            
            progress_bar.set_postfix({'training_loss': f'{total_train_loss/train_steps:.3f}'})
            
            # Save checkpoint and evaluate every save_interval steps
            if global_step % save_interval == 0:
                checkpoint_path = os.path.join(save_dir, 'latest_checkpoint.pt')
                torch.save({
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
                
                # Evaluation phase
                model.eval()
                total_val_loss = 0
                val_steps = 0
                val_samples = 0
                
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc="Validation"):
                        if val_samples >= max_eval_samples:
                            break
                            
                        texts = batch["text"]
                        indicators = batch["indicator"]
                        labels = batch["label"].to(device)
                        
                        outputs = model(texts, indicators)
                        outputs = outputs.squeeze(1)
                        
                        assert outputs.shape == labels.shape
                        loss = model.compute_mse_loss(outputs, labels)
                        
                        total_val_loss += loss.item()
                        val_steps += 1
                        val_samples += len(labels)
                
                avg_val_loss = total_val_loss / val_steps
                
                # Log validation metrics
                wandb.log({
                    "val_loss": avg_val_loss,
                    "global_step": global_step,
                })
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    model_save_path = os.path.join(save_dir, 'best_model.pt')
                    torch.save({
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': avg_val_loss,
                    }, model_save_path)
                    wandb.save(model_save_path)
                
                model.train()
        
        avg_train_loss = total_train_loss / train_steps
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "learning_rate": scheduler.get_last_lr()[0],
        })
        
        print(f"Epoch {epoch + 1}")
        print(f"Average training loss: {avg_train_loss:.3f}")
        
        scheduler.step()
    
    wandb.finish()

def main():
    # Load data
    with open("processed_data/data_for_train_bignews.json", "r") as f:
        data_for_train = json.load(f)
            
    # Split data into train and validation sets
    np.random.seed(42)
    indices = np.random.permutation(len(data_for_train))
    split = int(0.9 * len(data_for_train))
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    # train_data = [training_data[i] for i in train_indices if training_data[i][2].item() != 0]
    train_data = [data_for_train[i] for i in train_indices]
    val_data = [data_for_train[i] for i in val_indices]
    
    train_texts = [item["text"] for item in train_data]
    train_indicators = [item["indicator"] for item in train_data]
    train_labels = [item["bias_score"] for item in train_data]
    val_texts = [item["text"] for item in val_data]
    val_indicators = [item["indicator"] for item in val_data]
    val_labels = [item["bias_score"] for item in val_data]
    
    # Assert all training labels are >= 0
    assert all(label >= 0 for label in train_labels), "Found negative labels in training data!"
    assert all(label >= 0 for label in val_labels), "Found negative labels in validation data!"
    
    # Create datasets
    train_dataset = PoliticalBiasDataset(train_texts, train_indicators, train_labels)
    val_dataset = PoliticalBiasDataset(val_texts, val_indicators, val_labels)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize model
    model = PoliticalBiasCrossEncoder(model_name="microsoft/deberta-v3-large")
    print(model)
    # Train model
    train_cross_encoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=5,
        learning_rate=1e-6,
        save_dir="./checkpoints/bignews"
    )

if __name__ == "__main__":
    main()

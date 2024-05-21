import argparse
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, random_split

from tqdm import tqdm
import wandb

torch.random.manual_seed(1337)

from utils.dataset import Py150kDataset
from utils.tokenizer import BOS_ID, EOS_ID, PAD_ID
from config import load_config
from models import PyRNN, PyLSTM, PyTransformer

CHECKPOINT_PATH = Path("checkpoints/models")

# Things that can be "quantified" are handled by the config, things like architecture changes should be different classes (reduces boilerplate a tone)
# i.e. storing things like "prenorm" or "postnorm" in the config is nice but is effectively ignored by the __init__ 
def create_from_configuration(config_dict):
    print(config_dict)
    match config_dict["model_type"]:
        case "PyRNN":
            return PyRNN(**config_dict)
        case "PyLSTM":
            return PyLSTM(**config_dict)
        case "PyTransformer":
            return PyTransformer(**config_dict)
        case _:
            raise ValueError(f"Invalid model_type, got {config_dict['model_type']}")


def collate_fn(batch:list[torch.Tensor], max_len:int=2048):
    batch = [x[:max_len] for x in batch]
    batch = [
        torch.cat([torch.tensor([BOS_ID]), x, torch.tensor([EOS_ID])])
        for x in batch
    ]
    return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=PAD_ID)

def main(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {DEVICE}")

    config = load_config(args.config)
    print(config)
    config_dict = config.__dict__ # easy to use
    print(config_dict)

    # I don"t think validation loss matters as much when training generative models, if we manage to overfit on a large dataset then we are golden
    train_ds = Py150kDataset("train", config.tokenizer_name)
    val_ds = Py150kDataset("eval", config.tokenizer_name)
    train_extra_ds, val_ds, _ = random_split(val_ds, [0.85, 0.1, 0.05])
    train_ds = ConcatDataset([train_ds, train_extra_ds]) # 142.5k instead of 100k
    
    collate = partial(collate_fn, max_len=config_dict.get("context_window", 2048))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=collate, prefetch_factor=4, num_workers=8, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate, prefetch_factor=4, num_workers=8, persistent_workers=True)


    model = create_from_configuration(config_dict).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    wandb.init(
        project=config.wandb_project,
        config={
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "architecture": model.__class__.__name__,
            "n_training_examples": len(train_ds),
            "n_validation_examples": len(val_ds),
            "parameter_count": sum([p.numel() for p in model.parameters() if p.requires_grad])
        },
        group=config.wandb_group
    )

    model_path = CHECKPOINT_PATH / wandb.run.name
    model_path.mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(args.epochs):
        train_tqdm = tqdm(train_dl, desc=f"Epoch {epoch + 1}/{args.epochs} Training")
        total_train_loss = 0

        for i, batch in enumerate(train_tqdm):
            batch = batch.to(DEVICE)
            x = batch[..., :-1]
            y = batch[..., 1:]
            
            y_hat = model(x)
            if isinstance(y_hat, tuple): y_hat = y_hat[0]

            loss = criterion(y_hat.reshape(-1, config.vocab_size), y.reshape(-1))

            optim.zero_grad()
            loss.backward()
            optim.step()

            train_loss = loss.detach().cpu().numpy()
            total_train_loss += train_loss
            train_tqdm.set_postfix({"loss": f"{train_loss:.3f}"})

            if i % args.log_interval == 0:
                wandb.log({"train_loss": train_loss}, step=epoch * len(train_dl) + i)


        wandb.log({"avg_train_loss": total_train_loss / len(train_dl)}, step=(epoch+1) * len(train_dl)) # to get it on the same axis

        # Validation step
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            val_tqdm = tqdm(val_dl, desc=f"Epoch {epoch + 1}/{args.epochs} Validation")
            for val_batch in val_tqdm:
                val_batch = val_batch.to(DEVICE)
                x_val = val_batch[..., :-1]
                y_val = val_batch[..., 1:]

                y_hat = model(x_val)
                if isinstance(y_hat, tuple): y_hat = y_hat[0]

                loss = criterion(y_hat.reshape(-1, config.vocab_size), y_val.reshape(-1))
                val_loss = loss.detach().cpu().numpy()
                total_val_loss += val_loss
                val_tqdm.set_postfix({"val_loss": f"{val_loss:.3f}"})

        wandb.log({"avg_val_loss": total_val_loss / len(val_dl)}, step=(epoch+1) * len(train_dl)) # to get it on the same axis
        model.train()

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': total_train_loss / len(train_dl),
        }, model_path / f"epoch_{epoch + 1}.pt")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RNNs/Transformers for Python code generation")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log_interval", type=int, default=100, help="Number of batches between logging training status to Wandb")
    parser.add_argument("--continue_from", type=str, default=None, help="Path to checkpoint file to resume training from")
    args = parser.parse_args()
    main(args)


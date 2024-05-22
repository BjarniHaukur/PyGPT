import argparse
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, random_split

import wandb
import numpy as np
from tqdm import tqdm

torch.random.manual_seed(1337)

from utils.dataset import MemmapDataset
from utils.tokenizer import BPETokenizer, BOS_ID, EOS_ID, PAD_ID
from utils.metrics import bleu_score, syntax_error_score
from models import PyRNN, PyLSTM, PyTransformer, load_config, model_from_config

CHECKPOINT_PATH = Path("checkpoints/models")


def collate_fn(batch:list[torch.Tensor], max_len:int=2048):
    batch = [x[:max_len] for x in batch]
    batch = [
        torch.cat([torch.tensor([BOS_ID]), x, torch.tensor([EOS_ID])])
        for x in batch
    ]
    return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=PAD_ID)

def main(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE = "mps" if torch.backends.mps.is_available() else DEVICE
    
    print(f"Training on device: {DEVICE}")

    config = load_config(args.config)
    config_dict = config.__dict__ # easy to use

    # I don"t think validation loss matters as much when training generative models, if we manage to overfit on a large dataset then we are golden
    tokenizer = BPETokenizer.load(config.tokenizer_name)
    train_ds = MemmapDataset("train", config.tokenizer_name)
    val_ds = MemmapDataset("eval", config.tokenizer_name)
    train_extra_ds, val_ds, _ = random_split(val_ds, [0.85, 0.1, 0.05])
    train_ds = ConcatDataset([train_ds, train_extra_ds]) # 142.5k instead of 100k
    
    collate = partial(collate_fn, max_len=config_dict.get("context_window", args.seq_len))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=collate, prefetch_factor=4, num_workers=8, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate, prefetch_factor=4, num_workers=8, persistent_workers=True)

    model = model_from_config(config).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    if args.continue_from:
        checkpoint = torch.load(CHECKPOINT_PATH / args.continue_from, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        wandb.init(project=config.wandb_project, id=checkpoint['wandb_id'], resume="must")
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from checkpoint {args.continue_from}, starting at epoch {start_epoch}")
    else:
        wandb.init(
            project=config.wandb_project,
            config={
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "n_training_examples": len(train_ds),
                "n_validation_examples": len(val_ds),
                "parameter_count": sum([p.numel() for p in model.parameters() if p.requires_grad]),
                **config_dict
            },
            group=config.wandb_group
        )
        start_epoch = 0
        
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    model_path = CHECKPOINT_PATH / wandb.run.name
    model_path.mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_tqdm = tqdm(train_dl, desc=f"Epoch {epoch + 1}/{start_epoch + args.epochs} Training")
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

        model.eval()
        total_val_loss = 0.0
        total_val_perplex = 0.0
        with torch.no_grad():
            val_tqdm = tqdm(val_dl, desc=f"Epoch {epoch + 1}/{start_epoch + args.epochs} Validation")
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
                
                total_val_perplex += np.exp(val_loss)
                
        wandb.log({"avg_val_loss": total_val_loss / len(val_dl)}, step=(epoch+1) * len(train_dl)) # to get it on the same axis
        wandb.log({"avg_val_perplexity": total_val_perplex / len(val_dl)}, step=(epoch+1) * len(train_dl))
        
        batch = next(iter(val_dl)).to(DEVICE)
        B, L = batch.shape
        context = int(0.75 * L)
        x = batch[:, :context]
        y = batch[:, context:]
        y_hat = model.generate(B, max_len=L, starting_tokens=x)
        y_hat = y_hat[:, context:]
        avg_bleu_score = bleu_score(y.tolist(), y_hat)
                    
        gen = model.generate(B, max_len=200)
        programs = [tokenizer.detokenize(gen_seq) for gen_seq in gen]
        print(programs)
        avg_syntax_error_score = syntax_error_score(programs)
        
        wandb.log({"avg_bleu4_score": avg_bleu_score}, step=(epoch+1) * len(train_dl))
        wandb.log({"avg_syntax_error_score": avg_syntax_error_score}, step=(epoch+1) * len(train_dl))
        wandb.log({"generated_text": wandb.Html(tokenizer.color_text_html(programs[0]))}, step=(epoch+1) * len(train_dl))

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': total_train_loss / len(train_dl),
            'wandb_id': wandb.run.id,
            'config_name': args.config
        }, model_path / f"epoch_{epoch + 1}.pt")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RNNs/Transformers for Python code generation")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log_interval", type=int, default=100, help="Number of batches between logging training status to Wandb")
    parser.add_argument("--continue_from", type=str, default=None, help="Path to checkpoint file to resume training from")
    args = parser.parse_args()
    main(args)


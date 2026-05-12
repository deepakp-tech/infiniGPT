
import os
import urllib.request

import matplotlib.pyplot as plt
import torch
import tiktoken

from infini_gpt_config import INFINIGPT_CONFIG
from infini_transformer import InfiniTransformer
from infinigpt import InfiniGPT
from dataloader import InfiniGPT_dataloader


def calc_loss_batch(input_batch, target_batch, model, device):

    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)

    # CrossEntropyLoss for multi-class token prediction
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
    )

    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):

    total_loss = 0.0

    if len(data_loader) == 0:
        return float("nan")

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for batch_idx, (input_batch, target_batch) in enumerate(data_loader):

        if batch_idx >= num_batches:
            break

        loss = calc_loss_batch(
            input_batch,
            target_batch,
            model,
            device
        )

        total_loss += loss.item()

    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    Evaluate model on training and validation datasets.
    """

    model.eval()

    with torch.no_grad():

        train_loss = calc_loss_loader(
            train_loader,
            model,
            device,
            num_batches=eval_iter
        )

        val_loss = calc_loss_loader(
            val_loader,
            model,
            device,
            num_batches=eval_iter
        )

    model.train()

    return train_loss, val_loss


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    tokenizer,
):
    """
    Main training loop.
    """

    train_losses = []
    val_losses = []
    track_tokens_seen = []

    tokens_seen = 0
    global_step = -1

    for epoch in range(num_epochs):

        model.train()

        for input_batch, target_batch in train_loader:

            optimizer.zero_grad()

            loss = calc_loss_batch(
                input_batch,
                target_batch,
                model,
                device
            )

            # Backpropagation
            loss.backward()

            # Update model weights
            optimizer.step()

            # Count processed tokens
            tokens_seen += input_batch.numel()

            global_step += 1

            # Periodic evaluation
            if global_step % eval_freq == 0:

                train_loss, val_loss = evaluate_model(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    eval_iter
                )

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(
                    f"Epoch {epoch + 1} "
                    f"(Step {global_step:06d}) | "
                    f"Train Loss: {train_loss:.3f} | "
                    f"Validation Loss: {val_loss:.3f}"
                )

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """
    Plot training and validation losses.
    """

    fig, ax1 = plt.subplots()

    # Loss vs Epochs
    ax1.plot(epochs_seen, train_losses, label="Training Loss")
    ax1.plot(
        epochs_seen,
        val_losses,
        linestyle="-.",
        label="Validation Loss"
    )

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Secondary x-axis for tokens seen
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)

    ax2.set_xlabel("Tokens Seen")

    fig.tight_layout()


def load_text_data(url):
    """
    Download and load training text data.
    """

    with urllib.request.urlopen(url) as response:
        text_data = response.read().decode("utf-8")

    return text_data


def create_dataloaders(text_data, config):
    """
    Create train and validation dataloaders.
    """

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_loader = InfiniGPT_dataloader(
        text_data[:split_idx],
        batch_size=config["batch_size"],
        max_length=config["context_length"],
        stride=config["stride"],
        drop_last=True,
        shuffle=True,
    )

    val_loader = InfiniGPT_dataloader(
        text_data[split_idx:],
        batch_size=config["batch_size"],
        max_length=config["context_length"],
        stride=config["stride"],
        drop_last=False,
        shuffle=True,
    )

    return train_loader, val_loader


def main(config):

    # Reproducibility
    torch.manual_seed(123)

    # Device configuration
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    """
    Text Dataset
    """

    url = (
        "https://raw.githubusercontent.com/"
        "ArionDas/InfiniGPT/"
        "eb3abdc6eaf1d8f17be7e92e81fd641a710aae26/"
        "data/book_clean.txt"
    )

    text_data = load_text_data(url)

    """
    Model Initialization
    """

    model = InfiniGPT(INFINIGPT_CONFIG)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    """
    Dataloaders
    """

    train_loader, val_loader = create_dataloaders(
        text_data,
        config
    )

    """
    Tokenizer
    """

    tokenizer = tiktoken.get_encoding("gpt2")

    """
    Model Training
    """

    train_losses, val_losses, tokens_seen = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=config["num_epochs"],
        eval_freq=1,
        eval_iter=1,
        tokenizer=tokenizer,
    )

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":

    config = INFINIGPT_CONFIG

    train_losses, val_losses, tokens_seen, model = main(config)

    epochs_tensor = torch.linspace(
        0,
        config["num_epochs"],
        len(train_losses)
    )

    plot_losses(
        epochs_tensor,
        tokens_seen,
        train_losses,
        val_losses
    )

    plt.show()

    # Save trained model
    os.makedirs("model", exist_ok=True)

    torch.save(
        model.state_dict(),
        "model/sample_model.pth"
    )

    print("Model saved successfully.")
```

    

import os
import random
import numpy as np
import torch
import torch.nn as nn
import mlflow
import pandas as pd
from transformers import get_scheduler
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
import argparse
import dotenv
from huggingface_hub import login, HfApi
from sqlalchemy import create_engine

#import from custom modules
from sentiment_model.data import DataModule
from sentiment_model.model import load_model_and_tokenizer
from sentiment_model.training import train_epoch, validate_epoch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

#set_seed(cfg["seed"])

def setup_hf_auth() -> None:
    dotenv.load_dotenv()
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)
    user_info = HfApi().whoami()
    print(f"Logged in as: {user_info['name']}")
    print(f"Token has {user_info['auth']['type']} access.")

def get_db_engine():
    """
    Create postgres db engine object. This is assuming that a .env contains the necessary credentials.
    """
    dotenv.load_dotenv('./.env')

    DB_RE_HOST = os.getenv("DB_RE_HOST")
    DB_RE_DB_NAME = os.getenv("DB_RE_DB_NAME")
    DB_RE_PORT = os.getenv("DB_RE_PORT")
    DB_RE_USER = os.getenv("DB_RE_USER")
    DB_RE_PASSWORD = os.getenv("DB_RE_PASSWORD")

    if not all([DB_RE_HOST, DB_RE_DB_NAME, DB_RE_PORT, DB_RE_USER, DB_RE_PASSWORD]):
        raise ValueError("Missing one or more database credentials in .env")

    engine = create_engine(
        f"postgresql+psycopg2://{DB_RE_USER}:{DB_RE_PASSWORD}@{DB_RE_HOST}:{DB_RE_PORT}/{DB_RE_DB_NAME}"
    )
    #Report on credentials loaded
    print(f"DB host is: {DB_RE_HOST}")
    print(f"DB NAME is: {DB_RE_DB_NAME}")

    return engine
    
def setup_mlflow(
    experiment_name: str,
    tracking_uri: str = "file:./mlruns",
    artifact_location: str = None,
    ) -> None:
    mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        client.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location
        )
        print(f"Created experiment '{experiment_name}' → artifacts: {artifact_location or 'local'}")
    else:
        print(f"Using existing experiment '{experiment_name}' → artifacts: {experiment.artifact_location}")

    mlflow.set_experiment(experiment_name)


def save_checkpoint(model, epoch: int, val_accuracy: float, checkpoints_dir: str) -> None:
    path = os.path.join(checkpoints_dir, f"best_model_epoch{epoch}.pt")
    torch.save(model.state_dict(), path)
    print(f"Checkpoint saved: {path} (val_accuracy: {val_accuracy:.4f})")


def train(
    model_name="bert-base-uncased",
    experiment_name="bert-train-sentiment",
    device=None,
    tracking_uri="file:./mlruns",
    artifact_location=None,
    freeze_base=False,
    max_batches=None,
    epochs=1,
    ) -> pd.DataFrame:

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #this is hard coded, should be in config
    checkpoints_dir = "models/checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True) #create a local checkpoint dir

    best_val_accuracy = 0.0

    #Set up mlflow tracking
    setup_mlflow(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        artifact_location=artifact_location,
    )

    model, tokenizer = load_model_and_tokenizer(device, model_name, num_classes=2)
    
    #adding option to trainn only the head
    if freeze_base:
        for param in model.bert.parameters():
            param.requires_grad = False
        lr = 1e-3 #larger learning rate than full fine tuning
        print("Base BERT layers frozen — training classifier head only.")
    else:
        lr = 2e-5
        print("Full fine-tuning — all layers trainable.")

    data = DataModule(tokenizer)
    train_loader, val_loader, _ = data.loaders()

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    total_steps = len(train_loader) * epochs
    num_warmup_steps = int(0.06 * total_steps)
    print(train_loader.batch_size)
    print(len(train_loader))
    print(total_steps)

    #use hugging face cosine scheduler with built in warmup and options
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )

    #use automatic mixed precision: amp if GPU available
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else None

    # Collect metrics per epoch
    metrics_history = []

    with mlflow.start_run():
        mlflow.log_params({
            "model_name": model_name,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": 0.01,
            "scheduler": "cosine",
            "warmup_ratio": 0.06,
            "freeze_base": freeze_base,
        })

        for epoch in range(epochs):


            # ── Training ──────────────────────────────
            print("Training epoch ", epoch)
            avg_train_loss = train_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=loss_fn,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                epoch=epoch,
                epochs=epochs,
                max_batches=max_batches,
            )

            # ── Validation ────────────────────────────
            avg_val_loss, val_accuracy = validate_epoch(
                model=model,
                val_loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                max_batches=max_batches,
            )

            if val_accuracy > best_val_accuracy:
                save_checkpoint(
                    model=model,
                    epoch=epoch + 1,
                    val_accuracy=val_accuracy,
                    checkpoints_dir=checkpoints_dir,
                )
                best_val_accuracy = val_accuracy

            # Append epoch metrics to history
            metrics_history.append({
                "epoch":         epoch + 1,
                "train_loss":    round(avg_train_loss, 4),
                "val_loss":      round(avg_val_loss, 4),
                "val_accuracy":  round(val_accuracy, 4),
            })

            mlflow.log_metrics({
                "train_loss":   avg_train_loss,
                "val_loss":     avg_val_loss,
                "val_accuracy": val_accuracy,
            }, step=epoch)

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.4f}"
            )

        mlflow.pytorch.log_model(model, "model")

    # Build and return DataFrame
    metrics_df = pd.DataFrame(metrics_history).set_index("epoch")
    return metrics_df


def create_parser():
    parser = argparse.ArgumentParser(
        description="BERT Classification Training",
        epilog=(
            "Examples:\n"
            "  # Quick smoke-test with frozen base and 10 batches\n"
            "  python train.py --model_name bert-base-uncased --freeze_base --max_batches 10\n\n"
            "  # Full fine-tuning for 3 epochs with local MLflow\n"
            "  python train.py --model_name bert-base-uncased --number_epoch 3 \\\n"
            "      --experiment_name bert-train-sentiment \\\n"
            "      --tracking_uri file:./mlruns\n\n"
            "  # Full run logging to a remote MLflow server with GCS artifacts\n"
            "  python train.py --model_name bert-base-uncased --number_epoch 3 \\\n"
            "      --freeze_base --max_batches 10 \\\n"
            "      --experiment_name bert-train-sentiment \\\n"
            "      --tracking_uri http://mlflow-server:5000 \\\n"
            "      --artifact_location gs://my-bucket/mlflow/artifacts\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--freeze_base", action="store_true",
                        help="Freeze BERT layers and train classifier head only")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Limit batches per epoch for development (e.g. --max_batches 10)")
    parser.add_argument("--number_epoch",type=int,default=1)
    parser.add_argument("--experiment_name", type=str, default="bert-train-sentiment")
    parser.add_argument("--tracking_uri", type=str, default="file:./mlruns")
    parser.add_argument("--artifact_location", type=str, default=None,
                        help="GCS path e.g. gs://my-bucket/mlflow/artifacts")
    return parser


def main():
    setup_hf_auth()

    parser = create_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    metrics_df = train(
        model_name=args.model_name,
        experiment_name=args.experiment_name,
        device=device,
        tracking_uri=args.tracking_uri,
        artifact_location=args.artifact_location,
        freeze_base=args.freeze_base,
        max_batches=args.max_batches,
        epochs=args.number_epoch,
    )

    print("\nTraining Summary:")
    print(metrics_df.to_string())               # print full DataFrame


if __name__ == "__main__":
    main()
'''
python train.py \
--model_name bert-base-uncased \
--freeze_base \
--max_batches 100 \
--number_epoch 5 \
--experiment_name bert-train-sentiment \
--tracking_uri file:./mlruns 

python train.py \
--model_name bert-base-uncased \
--freeze_base \
--max_batches 10 \
--number_epoch 3 \
--experiment_name bert-train-sentiment \
--tracking_uri file:./mlruns 

python train.py \
--model_name bert-base-uncased \
--freeze_base \
--max_batches 10 \
--number_epoch 3 \
--experiment_name bert-train-sentiment \
--tracking_uri file:./mlruns \
--artifact_location gs://my-bucket/mlflow/artifacts 

'''
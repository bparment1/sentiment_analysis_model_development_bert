import os
import torch
import mlflow
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score, classification_report

from sentiment_model.data import DataModule
from sentiment_model.model import load_model_and_tokenizer
from train import setup_hf_auth, setup_mlflow


def evaluate(
    checkpoint_path: str,
    model_name: str = "bert-base-uncased",
    experiment_name: str = "bert-train-sentiment",
    device=None,
    tracking_uri: str = "file:./mlruns",
    artifact_location: str = None,
    max_batches: int = None,
) -> pd.DataFrame:
    """Evaluate a saved checkpoint on the IMDB test set.

    Loads model weights from checkpoint_path, runs inference on the held-out
    test split, logs metrics to MLflow, and returns a summary DataFrame.

    Args:
        checkpoint_path: Path to a .pt state-dict file produced by
            save_checkpoint() in train.py.
        model_name: HuggingFace model identifier used during training.
        experiment_name: MLflow experiment to log results into.
        device: torch.device to run inference on; auto-detected when None.
        tracking_uri: MLflow tracking server URI.
        artifact_location: Optional artifact store URI (e.g. gs://bucket/path).
        max_batches: Cap test batches — useful for quick smoke-tests.

    Returns:
        Single-row DataFrame with test accuracy, per-class F1, and macro
        precision/recall.

    Example:
        metrics = evaluate(
            checkpoint_path="checkpoints/best_model_epoch3.pt",
            model_name="bert-base-uncased",
            max_batches=10,
        )
        print(metrics)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    setup_mlflow(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        artifact_location=artifact_location,
    )

    model, tokenizer = load_model_and_tokenizer(device, model_name, num_classes=2)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    _, _, test_loader = DataModule(tokenizer).loaders()

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else None

    all_preds, all_labels = [], []

    with torch.inference_mode():
        for batch_num, batch in enumerate(test_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )

            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

            if max_batches and batch_num + 1 >= max_batches:
                break

    test_accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, target_names=["negative", "positive"], output_dict=True
    )

    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(classification_report(all_labels, all_preds, target_names=["negative", "positive"]))

    with mlflow.start_run(tags={"stage": "evaluation", "checkpoint": checkpoint_path}):
        mlflow.log_params({
            "model_name": model_name,
            "checkpoint_path": checkpoint_path,
        })
        mlflow.log_metrics({
            "test_accuracy":          test_accuracy,
            "test_f1_negative":       report["negative"]["f1-score"],
            "test_f1_positive":       report["positive"]["f1-score"],
            "test_f1_macro":          report["macro avg"]["f1-score"],
            "test_precision_macro":   report["macro avg"]["precision"],
            "test_recall_macro":      report["macro avg"]["recall"],
        })

    metrics = {
        "test_accuracy":        round(test_accuracy, 4),
        "test_f1_negative":     round(report["negative"]["f1-score"], 4),
        "test_f1_positive":     round(report["positive"]["f1-score"], 4),
        "test_f1_macro":        round(report["macro avg"]["f1-score"], 4),
        "test_precision_macro": round(report["macro avg"]["precision"], 4),
        "test_recall_macro":    round(report["macro avg"]["recall"], 4),
    }
    return pd.DataFrame([metrics])


def create_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate a BERT checkpoint on the test set",
        epilog=(
            "Examples:\n"
            "  # Evaluate best checkpoint on full test set\n"
            "  python evaluate.py --checkpoint_path checkpoints/best_model_epoch3.pt \\\n"
            "      --model_name bert-base-uncased \\\n"
            "      --experiment_name bert-train-sentiment \\\n"
            "      --tracking_uri file:./mlruns\n\n"
            "  # Quick smoke-test with 10 batches\n"
            "  python evaluate.py --checkpoint_path checkpoints/best_model_epoch3.pt \\\n"
            "      --model_name bert-base-uncased \\\n"
            "      --experiment_name bert-train-sentiment \\\n"
            "      --tracking_uri file:./mlruns \\\n"
            "      --max_batches 10\n\n"
            "  # Log to a remote MLflow server with GCS artifacts\n"
            "  python evaluate.py --checkpoint_path checkpoints/best_model_epoch3.pt \\\n"
            "      --model_name bert-base-uncased \\\n"
            "      --experiment_name bert-train-sentiment \\\n"
            "      --tracking_uri http://mlflow-server:5000 \\\n"
            "      --artifact_location gs://my-bucket/mlflow/artifacts\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the .pt checkpoint file")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--experiment_name", type=str, default="bert-train-sentiment")
    parser.add_argument("--tracking_uri", type=str, default="file:./mlruns")
    parser.add_argument("--artifact_location", type=str, default=None)
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Limit batches for development (e.g. --max_batches 10)")
    return parser


def main():
    setup_hf_auth()

    parser = create_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    metrics_df = evaluate(
        checkpoint_path=args.checkpoint_path,
        model_name=args.model_name,
        experiment_name=args.experiment_name,
        device=device,
        tracking_uri=args.tracking_uri,
        artifact_location=args.artifact_location,
        max_batches=args.max_batches,
    )

    print("\nEvaluation Summary:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()


'''
#quick test
python evaluate.py \
  --checkpoint_path models/checkpoints/best_model_epoch5.pt \
  --max_batches 10


#with track experiment name and full data

```
python evaluate.py \
  --checkpoint_path models/checkpoints/best_model_epoch5.pt \
  --model_name bert-base-uncased \
  --experiment_name bert-train-sentiment
```
'''
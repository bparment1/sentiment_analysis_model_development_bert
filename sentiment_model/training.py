import torch
from sklearn.metrics import accuracy_score

def train_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    loss_fn,
    device,
    use_amp: bool,
    amp_dtype,
    epoch: int,
    epochs: int,
    max_batches: int = None,
    ) -> float:
    model.train()
    total_loss = 0

    for batch_num, batch in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            loss = loss_fn(outputs.logits, batch["labels"])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if max_batches and batch_num + 1 >= max_batches:
            break

        if batch_num % 400 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Batch [{batch_num}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    batches_run = min(max_batches, len(train_loader)) if max_batches else len(train_loader)
    return total_loss / batches_run


def validate_epoch(
    model,
    val_loader,
    loss_fn,
    device,
    use_amp: bool,
    amp_dtype,
    max_batches: int = None,
) -> tuple[float, float]:
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []

    with torch.inference_mode():
        for val_batch_num, batch in enumerate(val_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                loss = loss_fn(outputs.logits, batch["labels"])

            val_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

            if max_batches and val_batch_num + 1 >= max_batches:
                break

            if val_batch_num % 400 == 0:
                print(
                    f"Val Batch [{val_batch_num}/{len(val_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

    batches_run = min(max_batches, len(val_loader)) if max_batches else len(val_loader)
    avg_val_loss = val_loss / batches_run
    val_accuracy = accuracy_score(all_labels, all_preds)
    return avg_val_loss, val_accuracy

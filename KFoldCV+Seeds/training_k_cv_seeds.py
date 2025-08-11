import os
import argparse
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import sys
import time

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--label_col', type=str, required=True, help='Name of the label column to use')
parser.add_argument('--label_id', type=str, required=True, help='Prefix number for the label')  
args = parser.parse_args()

LABEL_COL = args.label_col
LABEL_ID = args.label_id
N_SPLITS = 5
LABEL_FOLDER_NAME = f"{LABEL_ID}.{LABEL_COL}"
SAVE_DIR = LABEL_FOLDER_NAME
os.makedirs(SAVE_DIR, exist_ok=True)

class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w")  
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Combined log file
combined_log_path = os.path.join(SAVE_DIR, "run_log.txt")
sys.stdout = Logger(combined_log_path)
print(f"=== Script Started for LABEL_COL = '{LABEL_COL}' ===")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Running on device:", device)

# Parameters
MAX_LEN = 512
STRIDE = 256
EPOCHS = 15
LR = 1e-5

# Load dataset
df = pd.read_excel("/path/to/dataset", engine='openpyxl')
df = df[['personid','narrative',LABEL_COL]]
texts = df['narrative'].tolist()
labels = df[LABEL_COL].tolist()
ids = df['personid'].tolist()

# Hold out test set once (same for all seeds)
train_texts_full, test_texts, train_labels_full, test_labels, train_ids_full, test_ids = train_test_split(
    texts, labels, ids, test_size=0.1, stratify=labels, random_state=42)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Chunking function
def chunk_text(text, tokenizer, max_length=512, stride=256):
    tokens = tokenizer(text, return_tensors='pt', truncation=False)
    input_ids = tokens['input_ids'][0]
    attention_mask = tokens['attention_mask'][0]
    chunks, masks = [], []
    for i in range(0, len(input_ids), stride):
        end = i + max_length
        if i >= len(input_ids):
            break
        chunk_ids = input_ids[i:end]
        chunk_mask = attention_mask[i:end]
        if len(chunk_ids) < max_length:
            pad_len = max_length - len(chunk_ids)
            chunk_ids = torch.cat([chunk_ids, torch.zeros(pad_len, dtype=torch.long)])
            chunk_mask = torch.cat([chunk_mask, torch.zeros(pad_len, dtype=torch.long)])
        chunks.append(chunk_ids.unsqueeze(0))
        masks.append(chunk_mask.unsqueeze(0))
        if end >= len(input_ids):
            break
    return torch.cat(chunks), torch.cat(masks)

# Dataset class
class SlidingWindowDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.data = []
        for text, label in zip(texts, labels):
            input_ids, attention_mask = chunk_text(text, tokenizer, MAX_LEN, STRIDE)
            self.data.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label': torch.tensor(label, dtype=torch.float)
            })
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# Prepare test dataset
test_dataset = list(zip(test_ids, test_texts, test_labels))

# Define multiple seeds
seeds = [ 7, 46, 99, 123, 2024]
all_seed_results = []

best_model_info = {
    'f1': 0,
    'seed': None,
    'fold': None,
    'path': None
}

# Loop over each seed
for SEED in seeds:
    print(f"\n\n--- Starting Seed: {SEED} ---")
    
    # Set global seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Split full training data into folds using current seed
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_texts_full, train_labels_full)):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} (Seed: {SEED}) ---")
        
        x_train = [train_texts_full[i] for i in train_idx]
        y_train = [train_labels_full[i] for i in train_idx]
        x_val = [train_texts_full[i] for i in val_idx]
        y_val = [train_labels_full[i] for i in val_idx]

        train_dataset = SlidingWindowDataset(x_train, y_train, tokenizer)
        val_dataset = SlidingWindowDataset(x_val, y_val, tokenizer)

        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1).to(device)
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32).to(device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = AdamW(model.parameters(), lr=LR)

        best_f1 = 0.0
        model.train()
        for epoch in range(EPOCHS):
            start_time = time.time()
            total_train_loss = 0
            print(f"\nEpoch {epoch+1}/{EPOCHS}")
            model.train()
            for batch in train_dataset:
                optimizer.zero_grad()
                logits_list = []
                for input_id, mask in zip(batch['input_ids'], batch['attention_mask']):
                    input_id = input_id.unsqueeze(0).to(device)
                    mask = mask.unsqueeze(0).to(device)
                    output = model(input_ids=input_id, attention_mask=mask)
                    logits_list.append(output.logits.squeeze(0))
                avg_logits = torch.mean(torch.stack(logits_list), dim=0)
                label = batch['label'].unsqueeze(0).to(device)
                loss = loss_fn(avg_logits.view(-1), label.view(-1))
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(train_dataset)

            # Validation
            model.eval()
            val_loss = 0
            val_preds = []
            val_labels = []
            with torch.no_grad():
                for batch in val_dataset:
                    logits_list = []
                    for input_id, mask in zip(batch['input_ids'], batch['attention_mask']):
                        input_id = input_id.unsqueeze(0).to(device)
                        mask = mask.unsqueeze(0).to(device)
                        output = model(input_ids=input_id, attention_mask=mask)
                        logits_list.append(output.logits.squeeze(0))
                    avg_logits = torch.mean(torch.stack(logits_list), dim=0)
                    label = batch['label'].unsqueeze(0).to(device)
                    loss = loss_fn(avg_logits.view(-1), label.view(-1))
                    val_loss += loss.item()
                    prob = torch.sigmoid(avg_logits).item()
                    pred = 1 if prob > 0.5 else 0
                    val_preds.append(pred)
                    val_labels.append(label.item())
            avg_val_loss = val_loss / len(val_dataset)
            f1 = f1_score(val_labels, val_preds)
            precision = precision_score(val_labels, val_preds)
            recall = recall_score(val_labels, val_preds)
            epoch_time = time.time() - start_time

            # --- Logging ---
            print(f"Epoch Time: {epoch_time:.2f} sec")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val   Loss: {avg_val_loss:.4f}")
            print(f"Val   F1:   {f1:.4f}")
            print(f"Val   Precision: {precision:.4f}")
            print(f"Val   Recall:    {recall:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                print(f" Saving new best model (F1={f1:.4f})")
                save_path = os.path.join(SAVE_DIR, f"seed_{SEED}_fold_{fold+1}")
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)

                # Update global best model info
                if f1 > best_model_info['f1']:
                    best_model_info.update({
                        'f1': f1,
                        'seed': SEED,
                        'fold': fold + 1,
                        'path': save_path
                    })

        fold_metrics.append({
            'seed': SEED,
            'fold': fold + 1,
            'best_f1': best_f1,
            'precision': precision,
            'recall': recall
        })

    all_seed_results.extend(fold_metrics)

# Save all metrics
metrics_df = pd.DataFrame(all_seed_results)
metrics_df.to_csv(os.path.join(SAVE_DIR, "cv_metrics_all_seeds.csv"), index=False)

# Print final summary
avg_f1 = metrics_df['best_f1'].mean()
std_f1 = metrics_df['best_f1'].std()
print("\n\n--- Final Results Across All Seeds ---")
print(f"Avg F1: {avg_f1:.4f} ± {std_f1:.4f}")
print(f"Full Metrics Saved at: {os.path.join(SAVE_DIR, 'cv_metrics_all_seeds.csv')}")

# Print best model found
print("\n--- Best Model Summary ---")
print(f"Best F1 Score: {best_model_info['f1']:.4f}")
print(f"Found in Seed: {best_model_info['seed']}, Fold: {best_model_info['fold']}")
print(f"Model Path: {best_model_info['path']}")

# Evaluate best model on test set
print("\n--- Evaluating Best Model on Test Set ---")
model = BertForSequenceClassification.from_pretrained(best_model_info['path']).to(device)
model.eval()
all_preds, all_labels, all_ids = [], [], []

with torch.no_grad():
    for id_val, text, label in zip(test_ids, test_texts, test_labels):
        input_ids, attention_mask = chunk_text(text, tokenizer)
        logits_list = []
        for ids, mask in zip(input_ids, attention_mask):
            ids = ids.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            output = model(input_ids=ids, attention_mask=mask)
            logits_list.append(output.logits.squeeze(0))
        avg_logit = torch.mean(torch.stack(logits_list)).item()
        prob = torch.sigmoid(torch.tensor(avg_logit)).item()
        pred = 1 if prob > 0.5 else 0
        all_preds.append(pred)
        all_labels.append(label)
        all_ids.append(id_val)

# Classification report
print("\nClassification Report (Test Set):")
print(classification_report(all_labels, all_preds, target_names=["Class 0", "Class 1"]))

# Save predictions
results_df = pd.DataFrame({
    "id": all_ids,
    "text": test_texts,
    "true_label": all_labels,
    "predicted_label": all_preds
})
results_df.to_csv(os.path.join(SAVE_DIR, "test_predictions_final.csv"), index=False)

print("Training complete. Model, metrics, and predictions saved in:", SAVE_DIR)
print("=== Script Completed ===")
sys.stdout.log.close()
sys.stdout = sys.stdout.terminal
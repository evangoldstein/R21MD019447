import os
import argparse
import sys
import time
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

parser = argparse.ArgumentParser()
parser.add_argument('--label_col', type=str, required=True, help='Name of the label column to use')
parser.add_argument('--label_id', type=str, required=True, help='Prefix number for the label')  
args = parser.parse_args()

LABEL_COL = args.label_col
LABEL_ID = args.label_id

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

# Dataset path - modify this to point to your dataset
PATH = "/path/to/your/dataset.xlsx"

# Load dataset
df = pd.read_excel(PATH, engine='openpyxl')
df = df[['personid','narrative',LABEL_COL]]
id_col = df['personid']
text_col = df['narrative']
label = df[LABEL_COL]

# Train-val-test split
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df[LABEL_COL], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df[LABEL_COL], random_state=42)

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

# Prepare datasets
train_dataset = SlidingWindowDataset(train_df['narrative'], train_df[LABEL_COL], tokenizer)
val_dataset = SlidingWindowDataset(val_df['narrative'], val_df[LABEL_COL], tokenizer)
test_dataset = list(zip(test_df['personid'], test_df['narrative'], test_df[LABEL_COL]))

# Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)

# Weighted loss
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_df[LABEL_COL]), y=train_df[LABEL_COL])
pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32)
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

# Optimizer and device
optimizer = AdamW(model.parameters(), lr=LR)
model.to(device)

best_f1 = 0.0  # Track best F1 score

model.train()
for epoch in range(EPOCHS):
    epoch_start = time.time()
    total_train_loss = 0

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    model.train()

    # --- Training ---
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

    # --- Validation ---
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
    epoch_time = time.time() - epoch_start

    # --- Logging ---
    print(f"Epoch Time: {epoch_time:.2f} sec")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val   Loss: {avg_val_loss:.4f}")
    print(f"Val   F1:   {f1:.4f}")
    print(f"Val   Precision: {precision:.4f}")
    print(f"Val   Recall:    {recall:.4f}")

    # --- Save Best Model ---
    if f1 > best_f1:
        best_f1 = f1
        print(f" Saving new best model (F1={f1:.4f})")
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)


# Load best model
MODEL_DIR = SAVE_DIR
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

# Inference on test set
all_preds, all_labels, all_ids = [], [], []

with torch.no_grad():
    for id_val, text, label in test_dataset:
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

# Print classification report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["Class 0", "Class 1"]))

# Save predictions
results_df = pd.DataFrame({
    "id": all_ids,
    "text": test_df['narrative'].tolist(),
    "true_label": all_labels,
    "predicted_label": all_preds
})
results_df.to_csv(os.path.join(SAVE_DIR, "test_predictions.csv"), index=False)

print("Training complete. Model, metrics, and predictions saved in:", SAVE_DIR)

print("=== Script Completed ===")
sys.stdout.log.close()
sys.stdout = sys.stdout.terminal
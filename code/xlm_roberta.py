import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import nltk
from nltk.tokenize import sent_tokenize
import sys

# Download required NLTK resources
nltk.download('punkt')

# Detect environment (Colab or Kaggle)
IS_KAGGLE = 'kaggle' in os.environ.get('HOSTNAME', '') or os.path.exists('/kaggle')
print(f"Running in {'Kaggle' if IS_KAGGLE else 'Colab'} environment")

# Constants and Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
DIFFICULTY_LEVELS = ['easy', 'medium', 'hard']

# Environment-specific paths
if IS_KAGGLE:
    DATASET_DIR = '/kaggle/input/pan2025-dataset'  # Assumes dataset added via Kaggle input
    MODEL_SAVE_DIR = '/kaggle/working/models'
    OUTPUT_PATH = '/kaggle/working/outputs_xlm_roberta'
else:  # Colab
    DATASET_DIR = '/content/dataset'
    MODEL_SAVE_DIR = '/content/models'
    OUTPUT_PATH = '/content/outputs_xlm_roberta'
    DATASET_ZIP_PATH = '/content/pan2025_dataset.zip'

# Create directories
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Dataset handling for Colab
if not IS_KAGGLE:
    if not os.path.exists(DATASET_ZIP_PATH):
        print("Dataset zip file not found. Please upload pan2025_dataset.zip...")
        try:
            from google.colab import files
            uploaded = files.upload()
            if 'pan2025_dataset.zip' in uploaded:
                os.rename('pan2025_dataset.zip', DATASET_ZIP_PATH)
            else:
                raise FileNotFoundError("pan2025_dataset.zip was not uploaded.")
        except ImportError:
            raise RuntimeError("Upload functionality is only available in Colab.")
    
    if os.path.exists(DATASET_ZIP_PATH):
        print(f"Unzipping dataset from {DATASET_ZIP_PATH} to {DATASET_DIR}...")
        os.system(f"unzip -q {DATASET_ZIP_PATH} -d {DATASET_DIR}")
    else:
        raise FileNotFoundError(f"Could not find {DATASET_ZIP_PATH}.")

# Verify dataset directory
if os.path.exists(DATASET_DIR):
    print("Dataset directory exists. Listing contents:")
    print(os.listdir(DATASET_DIR))
else:
    raise FileNotFoundError(f"Dataset directory {DATASET_DIR} not found. Please ensure dataset is available.")

# Load tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

def load_dataset(base_path=DATASET_DIR):
    levels = ['easy', 'medium', 'hard']
    splits = ['train', 'validation']
    dataset = {}

    for level in levels:
        dataset[level] = {}
        for split in splits:
            split_path = os.path.join(base_path, level, split)
            documents = []

            if not os.path.exists(split_path):
                print(f"Warning: Path does not exist: {split_path}. Skipping...")
                continue

            try:
                files = sorted(os.listdir(split_path))
                for filename in files:
                    if filename.startswith('problem-') and filename.endswith('.txt'):
                        problem_id = filename.split('-')[1].split('.')[0]
                        txt_path = os.path.join(split_path, filename)
                        json_path = os.path.join(split_path, f'truth-problem-{problem_id}.json')

                        with open(txt_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                        sentences = sent_tokenize(text)

                        with open(json_path, 'r', encoding='utf-8') as f:
                            truth = json.load(f)
                        changes = truth.get('changes', [])
                        documents.append((sentences, changes, problem_id))
            except Exception as e:
                print(f"Error reading {split_path}: {str(e)}")
                continue

            dataset[level][split] = documents
            print(f"Loaded {len(documents)} documents for {level}/{split}")

    return dataset

class StyleChangeDataset(Dataset):
    def __init__(self, documents, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []
        self.labels = []
        self.problem_ids_with_offsets = []
        offset = 0

        for sentences, changes, problem_id in documents:
            for i in range(len(changes)):
                if i + 1 < len(sentences):
                    pair_text = sentences[i] + " </s> " + sentences[i + 1]
                    self.texts.append(pair_text)
                    self.labels.append(changes[i])
                    self.problem_ids_with_offsets.append((problem_id, i, offset + i))
            offset += len(changes)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

def prepare_datasets(documents, tokenizer, max_length=128):
    dataset = StyleChangeDataset(documents, tokenizer, max_length)
    return dataset, dataset.problem_ids_with_offsets

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Change (0)', 'Change (1)'],
                yticklabels=['No Change (0)', 'Change (1)'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(OUTPUT_PATH, f"{title.replace(' ', '_')}.png"))
    plt.show()
    return cm

def save_predictions_to_json(predictions, problem_ids_with_offsets, output_base_path, level, split):
    output_dir = os.path.join(output_base_path, level, split)
    os.makedirs(output_dir, exist_ok=True)
    pred_dict = {}
    for pred, (problem_id, _, offset) in zip(predictions, problem_ids_with_offsets):
        if problem_id not in pred_dict:
            pred_dict[problem_id] = []
        pred_dict[problem_id].append(int(pred))

    for problem_id, changes in pred_dict.items():
        solution = {"changes": changes}
        output_path = os.path.join(output_dir, f'solution-problem-{problem_id}.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(solution, f, indent=4)
        print(f"Saved: {output_path}")

def train_and_evaluate_xlm_roberta(train_dataset, val_dataset, val_problem_ids_with_offsets, level, output_base_path):
    pos_count = sum(1 for i in range(len(train_dataset)) if train_dataset[i]['labels'].item() == 1)
    neg_count = len(train_dataset) - pos_count
    pos_weight = neg_count / max(pos_count, 1)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=1, problem_type="regression")
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(DEVICE))

    best_val_f1 = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze()
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)

            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Validation"):
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze()
                if logits.dim() == 0:
                    logits = logits.unsqueeze(0)

                loss = loss_fn(logits, labels)
                val_loss += loss.item()

                preds = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()
                all_preds.extend(preds.tolist() if preds.ndim > 0 else [preds.item()])
                all_labels.extend(labels.cpu().numpy().tolist())

        val_loss /= len(val_loader)
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro')

        print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model_save_path = os.path.join(MODEL_SAVE_DIR, f"xlm_roberta_{level}_best.pt")
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved to {model_save_path}")

    # Final evaluation
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Final Evaluation"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze()
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)
            preds = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()
            all_preds.extend(preds.tolist() if preds.ndim > 0 else [preds.item()])
            all_labels.extend(labels.cpu().numpy().tolist())

    val_accuracy = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"\n{level} Level Final Metrics: Accuracy: {val_accuracy:.4f}, F1-Score: {val_f1:.4f}")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['No Change (0)', 'Change (1)']))

    cm = plot_confusion_matrix(all_labels, all_preds, f'Confusion Matrix - {level} Level')
    save_predictions_to_json(all_preds, val_problem_ids_with_offsets, output_base_path, level, 'validation')

    return model, all_labels, all_preds, cm

def main():
    print("Loading dataset...")
    dataset = load_dataset()

    all_val_labels, all_val_pred, all_cm = [], [], None
    for level in DIFFICULTY_LEVELS:
        print(f"\nProcessing {level} level...")
        try:
            train_docs = dataset.get(level, {}).get('train', [])
            val_docs = dataset.get(level, {}).get('validation', [])
            if not train_docs or not val_docs:
                print(f"Skipping {level} level due to missing data")
                continue

            train_dataset, _ = prepare_datasets(train_docs, tokenizer, MAX_LENGTH)
            val_dataset, val_problem_ids_with_offsets = prepare_datasets(val_docs, tokenizer, MAX_LENGTH)
            print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

            _, val_labels, val_pred, cm = train_and_evaluate_xlm_roberta(
                train_dataset, val_dataset, val_problem_ids_with_offsets, level, OUTPUT_PATH
            )
            all_val_labels.extend(val_labels)
            all_val_pred.extend(val_pred)
            all_cm = cm if all_cm is None else all_cm + cm
        except Exception as e:
            print(f"Error processing {level} level: {str(e)}")
            import traceback
            traceback.print_exc()

    if all_val_labels and all_val_pred:
        print("\nCombined Metrics Across All Levels:")
        print(classification_report(all_val_labels, all_val_pred, target_names=['No Change (0)', 'Change (1)']))
        plot_confusion_matrix(all_val_labels, all_val_pred, 'Combined Confusion Matrix - All Levels')

    tokenizer.save_pretrained(MODEL_SAVE_DIR)
    print(f"\nTraining completed! Models saved to: {MODEL_SAVE_DIR}, Outputs saved to: {OUTPUT_PATH}")
    zip_path = '/kaggle/working/xlm_roberta_models.zip'
    print(f"\nZipping model directory {MODEL_SAVE_DIR} for download...")
    shutil.make_archive(zip_path.replace('.zip', ''), 'zip', MODEL_SAVE_DIR)

if __name__ == "__main__":
    main()

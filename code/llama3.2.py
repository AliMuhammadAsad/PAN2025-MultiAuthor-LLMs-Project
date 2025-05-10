import os, json, torch
import numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

def load_dataset(base_path):
    levels = ['easy', 'medium', 'hard']
    splits = ['train', 'validation']
    dataset = {}

    for level in levels:
        dataset[level] = {}
        for split in splits:
            split_path = os.path.join(base_path, level, split)
            documents = []
            for filename in sorted(os.listdir(split_path)):
                if filename.startswith('problem-') and filename.endswith('.txt'):
                    problem_id = filename.split('.')[0]
                    txt_path = os.path.join(split_path, filename)
                    json_path = os.path.join(split_path, f'truth-{problem_id}.json')
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        sentences = [line.strip() for line in f.readlines() if line.strip()]
                    with open(json_path, 'r', encoding='utf-8') as f:
                        truth = json.load(f)
                        changes = truth['changes']
                    documents.append((sentences, changes, problem_id))
            dataset[level][split] = documents
    return dataset

def prepare_llama_data(documents, tokenizer, max_length=512):
    texts = []
    labels = []
    problem_ids_with_offsets = []
    offset = 0
    
    for sentences, changes, problem_id in documents:
        for i in range(len(changes)):
            pair_text = sentences[i] + " [SEP] " + sentences[i + 1]
            texts.append(pair_text)
            labels.append(changes[i])
            problem_ids_with_offsets.append((problem_id, i, offset + i))
        offset += len(changes)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    encodings = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    })
    
    return dataset, problem_ids_with_offsets

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Change (0)', 'Change (1)'],
                yticklabels=['No Change (0)', 'Change (1)'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    return cm

def save_predictions_to_json(predictions, problem_ids_with_offsets, output_base_path, level, split):
    output_dir = os.path.join(output_base_path, level, split)
    os.makedirs(output_dir, exist_ok=True)
    pred_dict = {}
    for pred, (problem_id, idx, offset) in zip(predictions, problem_ids_with_offsets):
        if problem_id not in pred_dict:
            pred_dict[problem_id] = []
        pred_dict[problem_id].append(int(pred))
    for problem_id, changes in pred_dict.items():
        solution = {"changes": changes}
        output_path = os.path.join(output_dir, f'solution-{problem_id}.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(solution, f, indent=4)
        print(f"Saved: {output_path}")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = (logits > 0).astype(int).flatten()
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return {'accuracy': accuracy, 'f1': f1}

def train_and_evaluate_llama(train_dataset, val_dataset, val_problem_ids_with_offsets, level, output_base_path, model_name):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        problem_type="regression"
    )
    
    # Update model config with pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Compute class weights for imbalanced data
    labels = train_dataset['labels']
    pos_weight = float((len(labels) - sum(labels)) / sum(labels))  # Convert to float
    
    # Custom Trainer to handle pos_weight in loss
    class WeightedBCELossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
            loss = loss_fn(logits.squeeze(), labels.float())
            return (loss, outputs) if return_outputs else loss
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/{level}",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"./logs/{level}",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )
    
    # Define trainer
    trainer = WeightedBCELossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train
    trainer.train()
    
    # Evaluate
    eval_results = trainer.evaluate()
    print(f"\n{level} Level Metrics:")
    print(f"Accuracy: {eval_results['eval_accuracy']:.4f}, F1-Score: {eval_results['eval_f1']:.4f}")
    
    # Predict
    predictions = trainer.predict(val_dataset)
    pred_labels = (predictions.predictions > 0).astype(int).flatten()
    
    # Classification report
    print("Classification Report:")
    print(classification_report(val_dataset['labels'], pred_labels, target_names=['No Change (0)', 'Change (1)']))
    
    # Plot confusion matrix
    cm = plot_confusion_matrix(val_dataset['labels'], pred_labels, f'Confusion Matrix - {level} Level')
    
    # Save predictions
    save_predictions_to_json(pred_labels, val_problem_ids_with_offsets, output_base_path, level, 'validation')
    
    return model, tokenizer, pred_labels, cm

dataset_dir = "../dataset"
print("Loading dataset...")
dataset = load_dataset(dataset_dir)

output_path = "../outputs_llama3.2"
model_name = "meta-llama/Llama-3.2-1B-Instruct"
max_length = 512  # Suitable for Llama's context window

all_val_labels = []
all_val_pred = []
all_cm = None

levels = ['easy', 'medium', 'hard']

for level in levels:
    print(f"Processing {level} level...")

    train_docs = dataset[level]['train']
    val_docs = dataset[level]['validation']

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    

    train_dataset, _ = prepare_llama_data(train_docs, tokenizer, max_length)
    val_dataset, val_problem_ids_with_offsets = prepare_llama_data(val_docs, tokenizer, max_length)
    
    # Train and evaluate
    model, tokenizer, val_pred, cm = train_and_evaluate_llama(
            train_dataset, val_dataset, val_problem_ids_with_offsets, level, output_path, model_name)
    
    all_val_labels.extend(val_dataset['labels'])
    all_val_pred.extend(val_pred)
    if all_cm is None:
        all_cm = cm
    else:
        all_cm += cm

print("\nCombined Metrics Across All Levels:")
print("Classification Report:")
print(classification_report(all_val_labels, all_val_pred, target_names=['No Change (0)', 'Change (1)']))
plot_confusion_matrix(all_val_labels, all_val_pred, 'Combined Confusion Matrix - All Levels')

# Save the model
model.save_pretrained("../models")
tokenizer.save_pretrained("../models")

print("Training and evaluation completed.")
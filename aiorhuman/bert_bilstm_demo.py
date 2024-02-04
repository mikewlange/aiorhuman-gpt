import argparse
import logging
import os
import pickle
import time
from IPython.display import display
import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                             precision_score, recall_score, f1_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (AdamW, BertModel, BertTokenizer,
                          get_linear_schedule_with_warmup)

from clearml import Task, OutputModel
from clearml_logger import ClearMLTaskHandler
import pandas as pd
import pickle
from clearml import Task, StorageManager

class CFG:
    # Device configuration
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ClearML Dataset IDs
    CLEAR_ML_KAGGLE_TRAIN_DATA = '24596ea241c34c6eb5013152a6122e48' #csv
    CLEAR_ML_AI_GENERATED_ESSAYS = '593fff56e3784e4fbfa4bf82096b0127' #pickle
    CLEAR_ML_AI_REWRITTEN_ESSAYS = '624315dd0e9b4314aa266654ebd71918' #pickle

    # Training configuration
    DEMO = True
    DATA_PATH = 'data'
    SCRATCH_PATH = 'scratch'
    ARTIFACTS_PATH = 'artifacts'
    BERT_MODEL = 'bert-base-uncased'
    
    CLEARML_PROJECT_NAME = 'Models - Text Classification'
    CLEARML_TASk_NAME = 'Train Bert bilstm'
    
run_name = f"run_{int(time.time())}"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
writer = SummaryWriter(log_dir=f'{CFG.SCRATCH_PATH}/logs/{run_name}')

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(texts, add_special_tokens=True, max_length=max_length, 
                                   padding='max_length', truncation=True, 
                                   return_attention_mask=True, return_tensors='pt').to(CFG.DEVICE)
        self.labels = torch.tensor(labels, dtype=torch.long).to(CFG.DEVICE)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['label'] = self.labels[idx]
        return item

class BERTBiLSTMClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes, dropout_rate=0.1, lstm_hidden_size=128, lstm_layers=2):
        super(BERTBiLSTMClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(self.bert.config.hidden_size, lstm_hidden_size, lstm_layers, 
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(lstm_hidden_size * 2, num_classes)  # *2 for bidirectional
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        lstm_output, (h_n, c_n) = self.lstm(sequence_output)
        pooled_output = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        x = self.dropout(pooled_output)
        x = self.relu(x)
        x = self.fc(x)
        return x

def train(model, data_loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    writer.add_scalar('Training Loss', avg_loss, epoch)
    print(f"Epoch {epoch} - Training loss: {avg_loss}")

def evaluate(model, data_loader, device, epoch, phase='Validation'):
    model.eval()
    predictions = []
    actual_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())

    accuracy = accuracy_score(actual_labels, predictions)
    precision = precision_score(actual_labels, predictions, average='binary', zero_division=1)
    recall = recall_score(actual_labels, predictions, average='binary')
    f1 = f1_score(actual_labels, predictions, average='binary', zero_division=1)
    auc = roc_auc_score(actual_labels, predictions)
    conf_matrix = confusion_matrix(actual_labels, predictions)

    sns.heatmap(conf_matrix, annot=True, fmt='d')
    plt.title(f'{phase} Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{phase}_confusion_matrix_epoch_{epoch}.png')
    plt.close()
    
    writer.add_scalar(f'{phase} Accuracy', accuracy, epoch)
    writer.add_scalar(f'{phase} Precision', precision, epoch)
    writer.add_scalar(f'{phase} Recall', recall, epoch)
    writer.add_scalar(f'{phase} F1 Score', f1, epoch)

    return accuracy, precision, recall, f1, auc, classification_report(actual_labels, predictions)

def parse_args():
    parser = argparse.ArgumentParser(description='BERT BiLSTM Text Classification')
    parser.add_argument('--bert-model-name', type=str, default='bert-base-uncased', help='BERT model name')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes for classification')
    parser.add_argument('--max-length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training and evaluation')
    parser.add_argument('--num-epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--dropout-rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--lstm-hidden-size', type=int, default=128, help='LSTM hidden layer size')
    parser.add_argument('--lstm-layers', type=int, default=2, help='Number of LSTM layers')
    return parser.parse_args()

def main():
    args = parse_args()s
    # Initialize ClearML task
    task = Task.init(project_name=CFG.CLEARML_PROJECT_NAME, task_name=CFG.CLEARMl_TASk_NAME, output_uri=True)
    task.connect(vars(args))

    if(CFG.DEMO):
        # or load the toy dataset
        df_combined = pd.read_csv('combined_data_toy.csv')
   
    else:
        return
 
    texts = df_combined['text'].str.lower().tolist()  # Lowercase for uncased BERT
    labels = df_combined['label'].tolist()
    
    tokenizer = BertTokenizer.from_pretrained(CFG.BERT_MODEL)
    # Split the data into training and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.3, random_state=42)
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, args.max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Initialize the model
    model = BERTBiLSTMClassifier(CFG.BERT_MODEL, args.num_classes, args.dropout_rate, args.lstm_hidden_size, args.lstm_layers)
    model.to(CFG.DEVICE)

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Training and Evaluation Loop
    for epoch in range(args.num_epochs):
        train(model, train_dataloader, optimizer, scheduler, CFG.DEVICE, epoch)
        accuracy, precision, recall, f1, auc, report = evaluate(model, val_dataloader, CFG.DEVICE, epoch)
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"precision: {precision:.4f}")
        print(f"recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")
        print(f"auc: {auc:.4f}")
        print(report)

    # Ensure the model is in evaluation mode before scripting
    #model.eval()

    # Was having touble with the torch jit. so we'll have to load our model with the objet class.
    model_path = 'bert_bilstm_model.pt'
    torch.save(model.state_dict(), model_path)

    output_model = OutputModel(task=task)
    output_model.update_weights(model_path)
    task.close()

    print("Training completed")

if __name__ == "__main__":
    main()




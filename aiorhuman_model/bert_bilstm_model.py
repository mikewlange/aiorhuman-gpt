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


def download_dataset_as_dataframe(dataset_id='593fff56e3784e4fbfa4bf82096b0127', file_name="ai_generated.pkl"):
    import pandas as pd
    # import Dataset from clearml
    from clearml import Dataset
    dataset = Dataset.get(dataset_id, only_completed=True)
    cached_folder = dataset.get_local_copy()
    for file_name in os.listdir(cached_folder):
        if file_name.endswith('.pkl'):
            file_path = os.path.join(cached_folder, file_name)
            dataframe = pd.read_pickle(file_path)
            return dataframe
    raise FileNotFoundError("No PKL file found in the dataset.")

def download_dataset_as_dataframe_csv(dataset_id='593fff56e3784e4fbfa4bf82096b0127', file_name="ai_generated_essays.csv"):
    import pandas as pd
    # import Dataset from clearml
    extension = file_name.split('.')[-1]
    from clearml import Dataset
    dataset = Dataset.get(dataset_id, only_completed=True)
    cached_folder = dataset.get_local_copy()
    for file_name in os.listdir(cached_folder):
        if file_name.endswith(extension):
            file_path = os.path.join(cached_folder, file_name)
            dataframe = pd.read_csv(file_path)
            return dataframe
        
class CFG:
    # Device configuration
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ClearML Dataset IDs
    #CLEAR_ML_TRAINING_DATASET_ID = '24596ea241c34c6eb5013152a6122e48'
    CLEAR_ML_KAGGLE_TRAIN_DATA = '24596ea241c34c6eb5013152a6122e48' #csv
    CLEAR_ML_AI_GENERATED_ESSAYS = '593fff56e3784e4fbfa4bf82096b0127' #pickle
    CLEAR_ML_AI_REWRITTEN_ESSAYS = '624315dd0e9b4314aa266654ebd71918' #pickle

    # Training configuration
    DEMO = 'False'
    DATA_PATH = 'data'
    SCRATCH_PATH = 'scratch'
    ARTIFACTS_PATH = 'artifacts'
    BERT_MODEL = 'bert-base-uncased'
    

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

    writer.add_scalar(f'{phase} Accuracy', accuracy, epoch)
    writer.add_scalar(f'{phase} Precision', precision, epoch)
    writer.add_scalar(f'{phase} Recall', recall, epoch)
    writer.add_scalar(f'{phase} F1 Score', f1, epoch)

    return accuracy, precision, recall, f1


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
    args = parse_args()
    # Initialize ClearML task
    task = Task.init(project_name='Models - Text Classification', task_name='train bert bilstm', output_uri=True)
    task.connect(vars(args))

    if(CFG.DEMO):
        # If you're folling for demo, download the two Grab the two training files here. or use the toy datasets from clearml
        #kaggle_training_data = 'https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset' # download_clearml_dataset_as_dataframe(CFG.CLEAR_ML_KAGGLE_TRAIN_DATA)
        #ai_generated_essays = 'https://www.kaggle.com/datasets/geraltrivia/llm-detect-gpt354-generated-and-rewritten-essays'
        # kaggle_training_data = kaggle_training_data.dropna(subset=['text'])
        # ai_generated_essays = ai_generated_essays.dropna(subset=['text'])
        # random_kaggle_training_data_0 = kaggle_training_data[kaggle_training_data['label'] == 0].sample(n=1000)[['text', 'label', 'source']]
        # random_kaggle_training_data_1 = kaggle_training_data[kaggle_training_data['label'] == 1].sample(n=400)[['text', 'label', 'source']]
        # random_ai_generated_essays = ai_generated_essays[ai_generated_essays['label'] == 1].sample(n=500)[['text', 'label', 'source']]
        # combined_data = pd.concat([random_kaggle_training_data_0, random_kaggle_training_data_1,random_ai_generated_essays])
        # df_combined = combined_data.reset_index(drop=True)
        # df_combined.drop_duplicates(inplace=True)

        # or load the toy dataset
        df_combined = pd.read_csv('combined_data_toy.csv')
   
    else:
        kaggle_training_data = download_dataset_as_dataframe_csv(dataset_id=CFG.CLEAR_ML_KAGGLE_TRAIN_DATA,file_name='train_raw_run_1.csv') # download_clearml_dataset_as_dataframe(CFG.CLEAR_ML_KAGGLE_TRAIN_DATA)
        ai_generated_essays = download_dataset_as_dataframe(dataset_id=CFG.CLEAR_ML_AI_GENERATED_ESSAYS,file_name='ai_generated.pkl')
        ai_rewritten_essays = download_dataset_as_dataframe(dataset_id=CFG.CLEAR_ML_AI_REWRITTEN_ESSAYS, file_name='ai_rewritten_essays.pkl')

        # Drop rows with missing values in 'text' column for each DataFrame
        kaggle_training_data = kaggle_training_data.dropna(subset=['text'])
        ai_generated_essays = ai_generated_essays.dropna(subset=['text'])
        ai_rewritten_essays = ai_rewritten_essays.dropna(subset=['text'])

        # Sample For Training Set. For demo. Use all data for production
        random_kaggle_training_data_0 = kaggle_training_data[kaggle_training_data['label'] == 0].sample(n=1000)[['text', 'label', 'source']]
        random_kaggle_training_data_1 = kaggle_training_data[kaggle_training_data['label'] == 1].sample(n=400)[['text', 'label', 'source']]
        random_ai_generated_essays = ai_generated_essays[ai_generated_essays['label'] == 1].sample(n=500)[['text', 'label', 'source']]
        random_ai_rewritten_essays = ai_rewritten_essays[ai_rewritten_essays['label'] == 1].sample(n=100)[['text', 'label', 'source']]
        
        combined_data = pd.concat([random_kaggle_training_data_0, random_kaggle_training_data_1,random_ai_generated_essays,random_ai_rewritten_essays])
        df_combined = combined_data.reset_index(drop=True)
        df_combined.drop_duplicates(inplace=True)
        
        # For Demo
        random_kaggle_training_data_0 = kaggle_training_data[kaggle_training_data['label'] == 0].sample(n=1000)[['text', 'label', 'source']]
        random_kaggle_training_data_1 = random_ai_generated_essays[random_ai_generated_essays['label'] == 1].sample(n=1000)[['text', 'label', 'source']]
        combined_data_toy = pd.concat([random_kaggle_training_data_0, random_kaggle_training_data_1,random_ai_generated_essays,random_ai_rewritten_essays])
        df_combined_toy = combined_data_toy.reset_index(drop=True)
        df_combined_toy.drop_duplicates(inplace=True)
        pd.save_csv('combined_data_toy.csv')
    
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
        evaluate(model, val_dataloader, CFG.DEVICE, epoch)

    # Ensure the model is in evaluation mode before scripting
    #model.eval()

    # Was having touble with the torch jit. so we'll have to load our model iwth the objet. no biggie but we want inference without reference to prig obj. script out. 
    model_path = 'bert_bilstm_model.pth'
    torch.save(model.state_dict(), model_path)

    output_model = OutputModel(task=task)
    output_model.update_weights(model_path)
    task.close()

    print("Training completed")

if __name__ == "__main__":
    main()




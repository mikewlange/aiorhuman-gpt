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
import optuna
# Set device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class CFG:
    # Device configuration
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ClearML Dataset IDs
    #CLEAR_ML_TRAINING_DATASET_ID = '24596ea241c34c6eb5013152a6122e48'
    CLEAR_ML_KAGGLE_TRAIN_DATA = '24596ea241c34c6eb5013152a6122e48' #csv
    CLEAR_ML_AI_GENERATED_ESSAYS = '593fff56e3784e4fbfa4bf82096b0127' #pickle
    CLEAR_ML_AI_REWRITTEN_ESSAYS = '624315dd0e9b4314aa266654ebd71918' #pickle

    # Training configuration
    
    DATA_PATH = 'data'
    SCRATCH_PATH = 'scratch'
    ARTIFACTS_PATH = 'artifacts'
    BERT_MODEL = 'bert-base-uncased'

# Model configuration
model_config = {
    'bert_model_name': CFG.BERT_MODEL,
    'num_classes': 2,
    'max_length': 128,
    'batch_size': 16,
    'num_epochs': 4,
    'num_trials': 2,
}

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

# Load data function
def load_data():

    kaggle_training_data = download_dataset_as_dataframe_csv(dataset_id=CFG.CLEAR_ML_KAGGLE_TRAIN_DATA,file_name='train_raw_run_1.csv') # download_clearml_dataset_as_dataframe(CFG.CLEAR_ML_KAGGLE_TRAIN_DATA)
    ai_generated_essays = download_dataset_as_dataframe(dataset_id=CFG.CLEAR_ML_AI_GENERATED_ESSAYS,file_name='ai_generated.pkl')
    ai_rewritten_essays = download_dataset_as_dataframe(dataset_id=CFG.CLEAR_ML_AI_REWRITTEN_ESSAYS, file_name='ai_rewritten_essays.pkl')

    # Drop rows with missing values in 'text' column for each DataFrame
    kaggle_training_data = kaggle_training_data.dropna(subset=['text'])
    ai_generated_essays = ai_generated_essays.dropna(subset=['text'])
    ai_rewritten_essays = ai_rewritten_essays.dropna(subset=['text'])

    # Sample For Training Set. For demo. Use all data for production
    random_kaggle_training_data_0 = kaggle_training_data[kaggle_training_data['label'] == 0].sample(n=2000)[['text', 'label', 'source']]
    random_kaggle_training_data_1 = kaggle_training_data[kaggle_training_data['label'] == 1].sample(n=900)[['text', 'label', 'source']]
    random_ai_generated_essays = ai_generated_essays[ai_generated_essays['label'] == 1].sample(n=1000)[['text', 'label', 'source']]
    random_ai_rewritten_essays = ai_rewritten_essays[ai_rewritten_essays['label'] == 1].sample(n=100)[['text', 'label', 'source']]
    
    combined_data = pd.concat([random_kaggle_training_data_0, random_kaggle_training_data_1,random_ai_generated_essays,random_ai_rewritten_essays])
    df_combined = combined_data.reset_index(drop=True)
    df_combined.drop_duplicates(inplace=True)
    
    
    texts = df_combined['text'].str.lower().tolist()  # Lowercase for uncased BERT
    labels = df_combined['label'].tolist()

    return texts, labels


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(texts, add_special_tokens=True, max_length=max_length, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt').to(device)
        self.labels = torch.tensor(labels, dtype=torch.long).to(device)

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
        self.lstm = nn.LSTM(self.bert.config.hidden_size, lstm_hidden_size, lstm_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(lstm_hidden_size * 2, num_classes)  # *2 for bidirectional
        self.relu = nn.ReLU()  # ReLU activation layer

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        lstm_output, (h_n, c_n) = self.lstm(sequence_output)
        pooled_output = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim = 1)
        x = self.dropout(pooled_output)
        x = self.relu(x)  # Apply ReLU activation
        x = self.fc(x)
        return x


# Custom BERT Classifier Model
class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes, dropout_rate=0.01, fc_layer_size=None):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout_rate)
        fc_layer_size = fc_layer_size if fc_layer_size is not None else self.bert.config.hidden_size
        self.fc = nn.Linear(self.bert.config.hidden_size, fc_layer_size)
        self.fc2 = nn.Linear(fc_layer_size, num_classes)
        self.relu = nn.ReLU()


    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.fc(pooled_output)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

texts, labels = load_data()
train_texts, temp_texts, train_labels, temp_labels = train_test_split(texts, labels, test_size=0.3, random_state=42)
val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5, random_state=42)

tokenizer = BertTokenizer.from_pretrained(model_config['bert_model_name'])
train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, model_config['max_length'])
val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, model_config['max_length'])
test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, model_config['max_length'])

train_dataloader = DataLoader(train_dataset, batch_size=model_config['batch_size'], shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=model_config['batch_size'])
test_dataloader = DataLoader(test_dataset, batch_size=model_config['batch_size'])


run_name = f"run_{int(time.time())}"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
writer = SummaryWriter(log_dir=f'{CFG.SCRATCH_PATH}/logs/bertmodel_custom/{run_name}')


def train(model, data_loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    leng = len(data_loader)
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        avg_loss = total_loss / leng
        #logger.info(f"Epoch {epoch} - Training loss: {avg_loss}")
        writer.add_scalar('Training Loss', avg_loss, epoch)

def evaluate(model, data_loader, device, epoch, phase='Validation'):

    model.eval()
    predictions = []
    actual_labels = []

    _labels = []
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

    #logger.info(f"Epoch {epoch} - {phase} Metrics - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
    writer.add_scalar(f'{phase} Accuracy', accuracy, epoch)
    writer.add_scalar(f'{phase} Precision', precision, epoch)
    writer.add_scalar(f'{phase} Recall', recall, epoch)
    writer.add_scalar(f'{phase} F1 Score', f1, epoch)

    return accuracy, precision, recall, f1, auc, classification_report(actual_labels, predictions)


# Optuna Hyperparameter Optimization
def objective(trial):
    # Suggest hyperparameters for training
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-5)
    batch_size = trial.suggest_int('batch_size', 16, 32)
    # Suggest hyperparameters for model architecture
    dropout_rate = trial.suggest_float('dropout_rate', 0.01, 0.1)
    fc_layer_size = trial.suggest_categorical('fc_layer_size', [32, 64])


    lstm_hidden_size = trial.suggest_categorical('lstm_hidden_size', [64, 128])# =128,
    lstm_layers=trial.suggest_int('lstm_layers', 2, 4)


    #model = BERTBiLSTMClassifier(model_config['bert_model_name'],model_config['num_classes'],dropout_rate,lstm_hidden_size )
    model = BERTBiLSTMClassifier(model_config['bert_model_name'], model_config['num_classes'], dropout_rate, fc_layer_size,lstm_layers)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    #total_steps = len(train_dataloader) / model_config['num_epochs'] / model_config['batch_size']
    total_steps = len(train_dataloader) * model_config['num_epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_val_auc = 0
    for epoch in tqdm(range(model_config['num_epochs']), desc='Epoch'):
        train(model, train_dataloader, optimizer, scheduler, device, epoch)
        accuracy, precision, recall, f1, auc, report = evaluate(model, val_dataloader, device, epoch)

        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"precision: {precision:.4f}")
        print(f"recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")
        print(f"auc: {auc:.4f}")
        print(report)

        if auc > best_val_auc:
            best_val_auc = auc
            best_params = {
                'learning_rate': learning_rate,
                'dropout_rate': dropout_rate,
                'fc_layer_size': fc_layer_size
            }
            torch.save(model.state_dict(), f"{CFG.SCRATCH_PATH}/bert_finetune_custom_{trial.number}.pt")

    torch.save(best_params, f"{CFG.SCRATCH_PATH}/best_trial_params.json")
    return best_val_auc


task = Task.init(project_name='Models - Text Classification - From Colab', task_name='train bert bilstm', output_uri=True)
cfg_dict = {key: value for key, value in CFG.__dict__.items() if not key.startswith('__')}
#args = parse_args()

task.connect(cfg_dict)
# Create a study object and optimize the objective function
bert_best_custom_study = optuna.create_study(direction='maximize', study_name='bert_best_custom_study')
bert_best_custom_study.optimize(objective, n_trials=model_config['num_trials'])

# Retrain model with best hyperparameters
best_trial = bert_best_custom_study.best_trial

#Load the model with the best trial
best_trial_params = bert_best_custom_study.best_trial.params
learning_rate = best_trial_params["learning_rate"]
dropout_rate = best_trial_params["dropout_rate"]
fc_layer_size = best_trial_params["fc_layer_size"]
lstm_hidden_size = best_trial_params["lstm_hidden_size"]
lstm_layers = best_trial_params["lstm_layers"]

# Pickle the tokenizer, study, and best model
with open(f'{CFG.SCRATCH_PATH}/custom_bert_tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

with open(f'{CFG.SCRATCH_PATH}/best_custom_model_study.pkl', 'wb') as f:
    pickle.dump(bert_best_custom_study, f)


#Initialize the best model with the optimal hyperparameters
best_model = BERTBiLSTMClassifier(model_config['bert_model_name'], model_config['num_classes'], dropout_rate, fc_layer_size,lstm_layers)
best_model.to(device)
#Set up optimizer and scheduler for the best model
optimizer = torch.optim.AdamW(best_model.parameters(), lr=learning_rate)
total_steps = len(train_dataloader) * model_config['num_epochs']
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

#Retrain the model with the best hyperparameters
for epoch in tqdm(range(model_config['num_epochs']), desc='Epoch'):
    train(best_model, train_dataloader, optimizer, scheduler, device, epoch)
    evaluate(best_model, val_dataloader, device, epoch)

#Save the retrained best model
torch.save(best_model.state_dict(), f"{CFG.SCRATCH_PATH}/bert_bilstm_model.pt")
output_model = OutputModel(task=task)
output_model.update_weights(f"{CFG.SCRATCH_PATH}/bert_bilstm_model.pt")

#Print best trial details
print("Best trial:")
print(f" Value: {best_trial.value:.4f}")
print(" Params: ")
for key, value in best_trial.params.items():
    print(f" {key}: {value}")

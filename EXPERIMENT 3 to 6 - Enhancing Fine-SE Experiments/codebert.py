import openpyxl
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import numpy as np
import time
import os
import torch.nn as nn
import torch.nn.functional as F
from openpyxl import load_workbook

# GLOBAL CONFIG
EPOCHS = 20
BATCH_SIZE_RATIO = 0.3
SEQUENCE_LEN = 20
LEARNING_RATE = 5e-4
OUTPUT = ''
MODEL = None
DYNAMIC_BATCH = True
BATCH_SIZE = None
WITHIN_PROJECT = None
MAE_RECORDS = []
MDAE_RECORDS = []

# define device
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
ROW_MAE, ROW_MMRE, ROW_PRED = 3, 4, 5


##############################################################################
# 1) DATA PROCESSING / LOADING
##############################################################################
def data_processing(file_pair, bs=None):
    """
    Loads the CSV, prepares train/val/test splits, tokenizes combined text,
    and sets up DataLoader objects.
    """
    global BATCH_SIZE, BATCH_SIZE_RATIO, DATA_PATH, WITHIN_PROJECT, DYNAMIC_BATCH

    fname = DATA_PATH + file_pair
    df = prepare_dataframe(fname)  # load columns

    (
        train_ex, train_text, train_labels,
        val_ex, val_text, val_labels,
        test_ex, test_text, test_labels
    ) = data_split(df)

    if bs is not None:
        BATCH_SIZE = bs
    elif DYNAMIC_BATCH:
        BATCH_SIZE = int(len(train_text) * BATCH_SIZE_RATIO)

    # Tokenization
    tokens_train = tokenization(train_text.tolist())
    tokens_val = tokenization(val_text.tolist())

    # Convert numeric columns to tensors
    train_seq = torch.tensor(tokens_train['input_ids'])
    train_ex = torch.tensor(np.array(train_ex))
    train_y = torch.tensor(train_labels.tolist()).type(torch.FloatTensor)
    # shape: [batch, 3 numeric] + [batch, token_ids up to SEQUENCE_LEN]
    train_seq = torch.cat((train_ex, train_seq), dim=1)
    train_dataloader = prepare_dataloader(train_seq, train_y, sampler_type='random', bs=BATCH_SIZE)

    val_seq = torch.tensor(tokens_val['input_ids'])
    val_ex = torch.tensor(np.array(val_ex))
    val_y = torch.tensor(val_labels.tolist()).type(torch.FloatTensor)
    val_seq = torch.cat((val_ex, val_seq), dim=1)
    val_dataloader = prepare_dataloader(val_seq, val_y, sampler_type='sequential', bs=BATCH_SIZE)

    # Prepare testing
    all_test_dataloader = []
    test_file_names = []

    tokens_test = tokenization(test_text.tolist())
    test_seq = torch.tensor(tokens_test['input_ids'])
    test_ex = torch.tensor(np.array(test_ex))
    test_seq = torch.cat((test_ex, test_seq), dim=1)
    test_y = torch.tensor(test_labels.tolist()).type(torch.FloatTensor)
    test_dataloader = prepare_dataloader(test_seq, test_y, sampler_type='sequential', bs=BATCH_SIZE)
    all_test_dataloader.append(test_dataloader)
    test_file_names.append(file_pair)

    return file_pair, train_dataloader, val_dataloader, all_test_dataloader


def prepare_dataframe(file_name):
    """
    Loads CSV with columns:
    [Assignee_count, Reporter_count, Creator_count, Summary, Description, Custom field (Story Points)]
    Then merges Summary + Description -> CombinedText
    """

    data = pd.read_csv(file_name)
    data = data.fillna("")  # fill missing with empty for text
    # Combine Summary + Description for CodeBERT
    data['CombinedText'] = (data['Issue Type'].astype(str) + " " + data['Version'].astype(str) + " " +
                            data['Summary'].astype(str) + " " + data['Description'].astype(str))

    # Adjust 'order' if your CSV has different column names or order
    final_order = [
        'Contributors', 'Developer_Rank', 'Developer_commits', 'Developer_created_MRs', 'Developer_modified_files',
        'Developer_commits_reviews', 'Developer_updated_MRs', 'Developer_fixed_defects', 'Developer_ARs',
        'Tester_detected_defects', 'Tester_ARs', 'Creator_ARs', 'Version_encoded',
        'CombinedText',
        'Custom field (Story Points)'
    ]
    data = data[final_order]
    # rename for clarity if you like
    data.rename(columns={
        'CombinedText': 'Text',
        'Custom field (Story Points)': 'Label'
    }, inplace=True)
    return data


def data_split(data):
    """
    Splits data into train(60%)/val(20%)/test(20%) within the same project.
    Returns numeric_ex, text_col, labels for each split.
    """
    train_val_split_point = int(len(data) * 0.6)
    val_test_split_point = int(len(data) * 0.8)

    train_ex = data.iloc[:train_val_split_point, 0:13]
    train_text = data['Text'][:train_val_split_point]
    train_labels = data['Label'][:train_val_split_point]

    val_ex = data.iloc[train_val_split_point:val_test_split_point, 0:13]
    val_text = data['Text'][train_val_split_point:val_test_split_point]
    val_labels = data['Label'][train_val_split_point:val_test_split_point]

    test_ex = data.iloc[val_test_split_point:, 0:13]
    test_text = data['Text'][val_test_split_point:]
    test_labels = data['Label'][val_test_split_point:]

    return train_ex, train_text, train_labels, val_ex, val_text, val_labels, test_ex, test_text, test_labels


##############################################################################
# 2) TOKENIZATION FOR CODEBERT
##############################################################################
def tokenization(text_list):
    """
    Tokenize the combined text with CodeBERT
    (which is "microsoft/codebert-base").
    We'll keep the same SEQUENCE_LEN logic as with standard BERT.
    """
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    return tokenizer(
        text_list,
        truncation=True,
        max_length=SEQUENCE_LEN,
        padding='max_length'
    )


def prepare_dataloader(seq, y, sampler_type, bs=None):
    global BATCH_SIZE

    if bs is not None:
        batch_size = bs
    else:
        batch_size = BATCH_SIZE

    tensor_dataset = TensorDataset(seq, y)
    if sampler_type == 'random':
        sampler = RandomSampler(tensor_dataset)
    else:
        sampler = SequentialSampler(tensor_dataset)

    dataloader = DataLoader(tensor_dataset, sampler=sampler, batch_size=batch_size)
    return dataloader


##############################################################################
# 3) MODEL DEFINITION (CODEBERT + Expert Features)
##############################################################################
class CodeBertForSequence(nn.Module):
    def __init__(self, sequence_len=20, numeric_feats=13, activation_fn=nn.ReLU(), dropout_prob=0.1, hidden_dim=50):
        super(CodeBertForSequence, self).__init__()
        self.sequence_len = sequence_len
        self.numeric_feats = numeric_feats
        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout_prob)

        # Add layer normalization for numeric features
        self.numeric_norm = nn.LayerNorm(numeric_feats, dtype=torch.float32)

        # Load CodeBERT
        self.codebert = AutoModel.from_pretrained("microsoft/codebert-base")
        for param in self.codebert.parameters():
            param.requires_grad = False
        for param in self.codebert.encoder.layer[-2:].parameters():
            param.requires_grad = True

        # Add layer norm after CodeBERT
        self.bert_norm = nn.LayerNorm(768, dtype=torch.float32)

        # Project from 768 to 3 dimensions
        self.hidden1 = nn.Sequential(
            nn.Linear(768, 3),
            nn.LayerNorm(3, dtype=torch.float32),
            self.activation_fn,
            self.dropout
        )

        # Combined features will be numeric_feats(13) + projected_features(3) = 16
        self.hidden2 = nn.Sequential(
            nn.Linear(16, hidden_dim),
            nn.LayerNorm(hidden_dim, dtype=torch.float32),
            self.activation_fn,
            self.dropout
        )

        # Final layer from hidden_dim to 1
        self.score = nn.Linear(hidden_dim, 1)  # Input dimension matches hidden_dim

    def forward(self, input_ids):
        # Extract numeric features and ensure Float type
        numeric_feats = input_ids[:, 0:13].float()
        numeric_feats = self.numeric_norm(numeric_feats)

        # Process text input
        code_input = input_ids[:, 13:].long()
        outputs_codebert = self.codebert(code_input)
        cls_embedding = outputs_codebert.last_hidden_state[:, 0, :]

        # Normalize CLS embedding
        cls_embedding = self.bert_norm(cls_embedding)

        # Project through hidden layers
        embed_3d = self.hidden1(cls_embedding)

        # Concatenate and process through final layers
        combined = torch.cat((numeric_feats, embed_3d), dim=1)  # Shape: [batch_size, 16]
        combined = self.hidden2(combined)  # Shape: [batch_size, hidden_dim]
        logit = self.score(combined)  # Shape: [batch_size, 1]

        return logit


##############################################################################
# 4) TRAIN/EVAL/TEST
##############################################################################
def train_eval_test(file_pair, train_dl, val_dl, test_dls, model, lr):
    global EPOCHS, MAE_RECORDS, MDAE_RECORDS, DEVICE

    if lr is not None:
        lr_rate = lr
    else:
        lr_rate = LEARNING_RATE

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate)
    total_steps = len(train_dl) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)
    print(f"Start training for {file_pair} with lr={lr_rate}, epochs={EPOCHS}...")

    min_eval_loss_epoch = [float('inf'), 0]
    time_records = []
    MAE_RECORDS, MDAE_RECORDS, MMRE_RECORDS, PRED_RECORDS = [], [], [], []
    start_time = time.time()
    loss_fct = nn.L1Loss()

    for e in range(EPOCHS):
        torch.cuda.empty_cache()
        print(">>> Epoch ", e)

        # TRAIN
        model.train()
        total_train_loss = 0
        for step, batch in enumerate(train_dl):
            b_input_ids, b_labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
            model.zero_grad()
            logits = model(b_input_ids)
            loss = loss_fct(logits, b_labels)
            total_train_loss += loss.item()
            loss.backward()

            # Add gradient clipping here, after backward() but before optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            del step, batch, b_input_ids, b_labels, logits, loss

        avg_train_loss = total_train_loss / len(train_dl)
        print(" Average training MAE loss: {0:.2f}".format(avg_train_loss))
        time_records.append(time.time() - start_time)

        # EVAL
        model.eval()
        total_eval_loss = 0
        for batch in val_dl:
            b_input_ids, b_labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
            with torch.no_grad():
                out = model(b_input_ids)
                loss = loss_fct(out, b_labels)
                total_eval_loss += loss.item()
            del batch, b_input_ids, b_labels, out, loss

        avg_eval_loss = total_eval_loss / len(val_dl)
        print(" Average eval MAE loss: {0:.2f}".format(avg_eval_loss))

        if avg_eval_loss <= min_eval_loss_epoch[0]:
            min_eval_loss_epoch[0] = avg_eval_loss
            min_eval_loss_epoch[1] = e

        del avg_eval_loss, total_eval_loss
        print("===============================")

        # TEST
        for test_dataloader in test_dls:
            predictions = []
            true_labels = []
            for batch in test_dataloader:
                batch = tuple(t.to(DEVICE) for t in batch)
                b_input_ids, b_labels = batch
                with torch.no_grad():
                    logits = model(b_input_ids)
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                predictions.append(logits)
                true_labels.append(label_ids)
            # calculate errors
            total_distance = 0
            total_mre = 0
            m = 0
            distance_records = []
            total_data_point = 0
            for i in range(len(predictions)):
                total_data_point += len(predictions[i])
            for i in range(len(predictions)):
                for j in range(len(predictions[i])):
                    distance = abs(predictions[i][j] - true_labels[i][j])
                    if (true_labels[i][j] > 0):
                        mre = abs(predictions[i][j] - true_labels[i][j]) / true_labels[i][j]
                    else:
                        mre = (abs(predictions[i][j] - true_labels[i][j]) + 1) / (true_labels[i][j] + 1)
                    if mre < 0.5:
                        m += 1
                    total_mre += mre
                    total_distance += distance
                    distance_records.append(distance)
            MAE = total_distance / total_data_point
            MMRE = total_mre / total_data_point
            MdAE = np.median(np.array(distance_records))
            PRED = m / total_data_point
            MAE_RECORDS.append(MAE)
            MDAE_RECORDS.append(MdAE)
            MMRE_RECORDS.append(MMRE)
            PRED_RECORDS.append(PRED)

            global OUTPUT
            OUTPUT += 'Epochs ' + str(e) + '\n'
            OUTPUT += 'MAE: ' + str(MAE) + '\n'
            OUTPUT += 'MdAE: ' + str(MdAE) + '\n'
            OUTPUT += 'MMRE: ' + str(MMRE) + '\n'
            OUTPUT += 'PRED: ' + str(PRED) + '\n\n'
            print('MAE: ', MAE)
            print('MdAE: ', MdAE)
            print('MMRE: ', MMRE)
            print('PRED: ', PRED)

    OUTPUT += f"{MAE_RECORDS[min_eval_loss_epoch[1]]}\n"
    OUTPUT += f"{MMRE_RECORDS[min_eval_loss_epoch[1]]}\n"
    OUTPUT += f"{PRED_RECORDS[min_eval_loss_epoch[1]]}\n"
    OUTPUT += f"training time: {time_records[min_eval_loss_epoch[1]]}\n"
    OUTPUT += f"Epochs: {min_eval_loss_epoch[1]}\n"
    global BATCH_SIZE
    OUTPUT += f"batch size: {BATCH_SIZE}"
    print("Training completed.")
    print(f"Best epoch: {min_eval_loss_epoch[1]} with eval loss: {min_eval_loss_epoch[0]:.2f}")


def define_params_optuna(file):
    if 'mesos' in file.lower():
        model_lr = 0.0008784969754294248
        batch_size = 8
        activation_fn = nn.Tanh()
        dropout_prob = '0.1627421470175328'
        hidden_dim = 64
    if 'usergrid' in file.lower():
        model_lr = 0.00030288052820257125
        batch_size = 8
        activation_fn = nn.LeakyReLU()
        dropout_prob = 0.13494423654862897
        hidden_dim = 64
    if 'data_management' in file.lower():
        model_lr = 0.0005503245217026587
        batch_size = 16
        activation_fn = nn.Tanh()
        dropout_prob = 0.11559239619903759
        hidden_dim = 32

    return model_lr, batch_size, activation_fn, dropout_prob, hidden_dim


# def define_params_grid(file):
#     if 'mesos' in file.lower():
#         latent_dim = 5
#         ae_lr = 1e-05
#         lr = 0.001
#         ae_epochs = 20
#         batch_size = 32
#     if 'usergrid' in file.lower():
#         latent_dim = 5
#         ae_lr = 0.0001
#         lr = 0.001
#         ae_epochs = 50
#         batch_size = 8
#     if 'data_management' in file.lower():
#         latent_dim = 10
#         ae_lr = 0.00045973705603571474
#         lr = 0.0008784969754294248
#         ae_epochs = 50
#         batch_size = 8
#
#     return latent_dim, ae_lr, lr, ae_epochs, batch_size


def main(result_path, files, hyper_params):
    global MODEL

    for file in files:
        model_lr, batch_size, activation_fn, dropout_prob, hidden_dim = (
            (define_params_optuna(file))
            if hyper_params
            else (LEARNING_RATE, None, nn.ReLU(), 0.1, 50)
        )

        MODEL = CodeBertForSequence().to(DEVICE)
        MODEL.activation_fn = activation_fn
        MODEL.dropout_prob = dropout_prob
        MODEL.hidden_dim = hidden_dim

        file_pair, train_dataloader, val_dataloader, all_test_dataloader = data_processing(
            file_pair=file, bs=batch_size)
        train_eval_test(file_pair, train_dataloader, val_dataloader, all_test_dataloader, MODEL, lr=model_lr)

        # Cleanup
        del MODEL
        torch.cuda.empty_cache()

        global OUTPUT
        file_result_path = result_path + file[:-4] + '.txt'
        os.makedirs(os.path.dirname(file_result_path), exist_ok=True)

        if not os.path.exists(file_result_path):
            with open(file_result_path, 'w+') as f:
                f.writelines(OUTPUT)
                OUTPUT = ""
        else:
            with open(file_result_path, 'w') as f:
                f.writelines(OUTPUT)
                OUTPUT = ""


if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("Device name:", torch.cuda.get_device_name(0))

    # define files to be used
    DATA_PATH = r'data/'
    DATA_FILES = ['MESOS', 'USERGRID', 'DATA_MANAGEMENT']
    FILE_TYPE = '.csv'
    train_test_file_paths = []

    for folder in DATA_FILES:
        folder_files = [f for f in os.listdir(os.path.join(DATA_PATH, folder)) if f.endswith(FILE_TYPE)]
        folder_files.sort(key=str.lower)

        for f in folder_files:
            train_test_file_paths.append(f"{folder}/{f}")

    optuna = True
    hyper_param = True

    if hyper_param:
        main(result_path='results/codeBERT/hyper_param/', files=train_test_file_paths, hyper_params=hyper_param)
    else:
        main(result_path='results/codeBERT/default/', files=train_test_file_paths, hyper_params=hyper_param)

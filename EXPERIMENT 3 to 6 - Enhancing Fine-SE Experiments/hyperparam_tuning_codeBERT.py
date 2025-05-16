import os
import json
import time
import optuna
from functools import partial

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup

###############################################################################
# GLOBAL CONFIG
###############################################################################
projectnum = 2
EPOCHS = 20  # default epochs for the model
BATCH_SIZE_RATIO = 0.3
SEQUENCE_LEN = 20  # tokens for summary & description
LEARNING_RATE = 3e-4  # default model LR; will be overridden by tuner
OUTPUT = ''
MODEL = None
DYNAMIC_BATCH = True
BATCH_SIZE = 16
WITHIN_PROJECT = True
MAE_RECORDS, MMRE_RECORDS, PRED_RECORDS, MDAE_RECORDS = [], [], [], []

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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


###############################################################################
# 4) TRAIN/EVAL/TEST
###############################################################################
def train_eval_test(file_pair, train_dl, val_dl, test_dl, model,
                    epochs=EPOCHS, lr=LEARNING_RATE):
    """
    Trains the model on the given train_dl, evaluates on val_dl,
    and also prints performance on test_dl each epoch. Returns
    the best validation loss (MAE) as a measure for Optuna.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = int(len(train_dl)) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

    print(f"Start training for {file_pair} with lr={lr}, epochs={epochs}...")
    min_eval_loss = float('inf')
    best_epoch = 0
    time_records = []
    start_time = time.time()
    loss_fct = nn.L1Loss()

    for e in range(epochs):
        torch.cuda.empty_cache()
        print(">>> Epoch ", e)
        model.train()
        total_train_loss = 0

        for step, batch in enumerate(train_dl):
            b_input_ids, b_labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
            model.zero_grad()
            logits = model(b_input_ids)
            loss = loss_fct(logits, b_labels)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

            del step, batch, b_input_ids, b_labels, logits, loss

        avg_train_loss = total_train_loss / len(train_dl)
        print(" Average training MAE loss: {0:.2f}".format(avg_train_loss))

        time_records.append(time.time() - start_time)
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

        if avg_eval_loss < min_eval_loss:
            min_eval_loss = avg_eval_loss
            best_epoch = e

        print("===============================")

        # Evaluate on test each epoch
        for test_dataloader in test_dl:
            predictions = []
            true_labels = []
            for batch in test_dataloader:
                batch = tuple(t.to(DEVICE) for t in batch)
                b_input_ids, b_labels = batch
                with torch.no_grad():
                    pred = model(b_input_ids)
                predictions.append(pred.cpu().numpy())
                true_labels.append(b_labels.cpu().numpy())

            total_distance = 0
            total_mre = 0
            m = 0
            distance_records = []
            total_data_point = sum(len(pred) for pred in predictions)
            for i in range(len(predictions)):
                for j in range(len(predictions[i])):
                    distance = abs(predictions[i][j] - true_labels[i][j])
                    if true_labels[i][j] > 0:
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

            global OUTPUT
            OUTPUT += f"Epochs {e}\nMAE: {MAE}\nMdAE: {MdAE}\nMMRE: {MMRE}\nPRED: {PRED}\n\n"
            print("MAE: ", MAE)
            print("MdAE: ", MdAE)
            print("MMRE: ", MMRE)
            print("PRED: ", PRED)

    OUTPUT += f"Best eval loss: {min_eval_loss:.4f}\n"
    OUTPUT += f"Best epoch: {int(best_epoch)}\n"
    OUTPUT += f"training time: {time_records[int(best_epoch)]}\n"
    OUTPUT += f"batch size: {BATCH_SIZE}\n"
    print("Training completed.")
    print(f"Best epoch: {best_epoch} with eval loss: {min_eval_loss:.2f}")

    # Return min eval loss so Optuna can minimize it
    return min_eval_loss


###############################################################################
# 5) OPTUNA INTEGRATION
###############################################################################
def objective(trial, file_pair, result_path):
    """
    Optuna objective for a single dataset/file.
    It samples hyperparameters, runs data_processing,
    trains/evaluates the model, and returns the val_loss.
    """
    # Hyperparameter Suggestions
    model_lr = trial.suggest_loguniform("model_lr", 1e-5, 1e-3)
    bs = trial.suggest_categorical("batch_size", [8, 16, 32])
    activation_fn = trial.suggest_categorical("activation_fn", ["relu", "tanh", "leaky_relu"])  # New
    dropout_prob = trial.suggest_float("dropout_prob", 0.1, 0.5)  # New
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 50, 64])  # New

    # Adjust activation function mapping
    activation_map = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU()
    }
    activation_function = activation_map[activation_fn]

    # -------- 2. Build data loaders --------
    file_pair_out, train_dl, val_dl, test_dls = data_processing(file_pair=file_pair, bs=bs)

    # -------- 3. Instantiate the model --------
    model = CodeBertForSequence(
        activation_fn=activation_function,
        dropout_prob=dropout_prob,
        hidden_dim=hidden_dim
    ).to(DEVICE)

    # -------- 4. Train & evaluate --------
    val_loss = train_eval_test(
        file_pair_out, train_dl, val_dl, test_dls,
        model,
        epochs=EPOCHS,
        lr=model_lr
    )

    # Optional: store partial logs for each trial
    file_result_path = result_path + file_pair[:-4] + '_trial.txt'
    os.makedirs(os.path.dirname(file_result_path), exist_ok=True)
    with open(file_result_path, 'a') as f:
        f.write(f"\nTrial {trial.number} => val_loss={val_loss:.4f}, params={trial.params}\n")

    return val_loss


def tune_hyperparams(files, result_path, n_trials=10):
    """
    Iterates over each file, runs an Optuna study,
    and returns a dictionary of the best hyperparams found.
    """
    best_params_dict = {}

    for file_pair in files:
        print(f"\n===== Tuning hyperparameters for: {file_pair} =====")

        # Lock in the file_pair so Optuna only passes trial
        objective_for_file = partial(objective,
                                     file_pair=file_pair,
                                     result_path=result_path)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective_for_file, n_trials=n_trials)

        best_params_dict[file_pair] = {
            "best_value": study.best_value,
            "best_params": study.best_params
        }

        print(f"Best val_loss = {study.best_value:.4f}")
        print(f"Best params   = {study.best_params}")

    return best_params_dict


###############################################################################
# 6) MAIN
###############################################################################
def main(result_path, files):
    """
    If you still want to run your original pipeline without tuning,
    you can keep this method. Otherwise, you can comment it out
    or adapt it to run the best hyperparams from the tuner.
    """
    global MODEL
    for file in files:
        MODEL = CodeBertForSequence().to(DEVICE)
        file_pair, train_dl, val_dl, test_dls = data_processing(file_pair=file)
        train_eval_test(file_pair, train_dl, val_dl, test_dls, MODEL)
        del MODEL
        torch.cuda.empty_cache()

        global OUTPUT
        file_result_path = result_path + file[:-4] + '.txt'
        os.makedirs(os.path.dirname(file_result_path), exist_ok=True)
        with open(file_result_path, 'w+') as f:
            f.writelines(OUTPUT)
        OUTPUT = ""
        global projectnum
        projectnum += 1


if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        print("Device name:", torch.cuda.get_device_name(0))

    torch.cuda.manual_seed_all(0)
    DATA_PATH = r'data/'
    DATA_FILES = ['MESOS', 'USERGRID', 'DATA_MANAGEMENT']
    FILE_TYPE = '.csv'
    train_test_file_paths = []
    for folder in DATA_FILES:
        folder_files = [f for f in os.listdir(os.path.join(DATA_PATH, folder)) if f.endswith(FILE_TYPE)]
        folder_files.sort(key=str.lower)
        for f in folder_files:
            train_test_file_paths.append(f"{folder}/{f}")

    # -------------------------------------------------------------------------
    #  A) If you want to do hyperparameter tuning, call:
    # -------------------------------------------------------------------------
    result_path = 'results/hyper_param_tuning/codeBERT/'
    best_params = tune_hyperparams(
        files=train_test_file_paths,
        result_path=result_path,
        n_trials=10  # adjust as desired
    )
    print("\nAll done! Best params found:")
    for file_name, info in best_params.items():
        print(f"{file_name} -> val_loss: {info['best_value']:.4f}, hyperparams: {info['best_params']}")

    # -------------------------------------------------------------------------
    # B) If you just want to run the existing pipeline WITHOUT tuning, call:
    # -------------------------------------------------------------------------
    # main(result_path='results/autoencoders/', files=train_test_file_paths)

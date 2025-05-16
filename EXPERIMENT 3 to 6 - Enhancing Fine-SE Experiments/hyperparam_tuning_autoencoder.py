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
EPOCHS = 30  # default epochs for the model
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


###############################################################################
# AUTOENCODER FOR NUMERIC FEATURES
###############################################################################
class NumericAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(NumericAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent


def autoencoder_transform_train_val_test(X_train, X_val, X_test, latent_dim=5,
                                         num_epochs=50, ae_lr=3e-4,
                                         save_path="data/AUTOENCODER_MODEL"):
    input_dim = X_train.shape[1]
    autoencoder = NumericAutoencoder(input_dim, latent_dim).to(DEVICE)

    # Load pre-trained weights if they exist; otherwise train from scratch
    # if os.path.exists(save_path):
    #    print("🔄 Loading pre-trained autoencoder weights...")
    #    autoencoder.load_state_dict(torch.load(save_path + '/autoencoder.pth'))
    # else:
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=ae_lr)
    loss_fn = nn.MSELoss()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float).to(DEVICE)
    dataset = TensorDataset(X_train_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(num_epochs):
        autoencoder.train()
        for batch in dataloader:
            batch = batch[0]
            optimizer.zero_grad()
            reconstruction, _ = autoencoder(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()

    #if not os.path.exists(save_path):
    #    os.makedirs(save_path)

    #torch.save(autoencoder.state_dict(), save_path + '/autoencoder.pth')


    # Transform numeric features to their latent representation
    autoencoder.eval()
    with torch.no_grad():
        X_train_latent = autoencoder.encoder(torch.tensor(X_train, dtype=torch.float).to(DEVICE)).cpu().numpy()
        X_val_latent = autoencoder.encoder(torch.tensor(X_val, dtype=torch.float).to(DEVICE)).cpu().numpy()
        X_test_latent = autoencoder.encoder(torch.tensor(X_test, dtype=torch.float).to(DEVICE)).cpu().numpy()

    return X_train_latent, X_val_latent, X_test_latent


###############################################################################
# 1) DATA PROCESSING
###############################################################################
def data_processing(file_pair,
                    latent_dim=5,
                    ae_epochs=50,
                    ae_lr=3e-4,
                    lr=LEARNING_RATE,
                    bs=None):
    """
    Prepares data for training/validation/testing.
    This includes:
    - Numeric autoencoder transformation
    - Text tokenization for summary & description
    - Combining numeric + textual features into a single tensor
    - Returning DataLoaders for train, val, and test
    """
    global BATCH_SIZE, BATCH_SIZE_RATIO, DATA_PATH

    fname = DATA_PATH + file_pair
    df = prepare_dataframe(fname)

    df["AugSummary"] = df["Issue Type"].astype(str) + " " + \
                       df["Version"].astype(str) + " " + \
                       df["Summary"].astype(str)

    train_df, val_df, test_df = data_split(df)

    excluded = ["AugSummary", "Description", "Label", "Issue Type", "Version", "Summary"]
    numeric_cols = [col for col in df.columns if col not in excluded]

    train_numeric = train_df[numeric_cols].values.astype(np.float32)
    val_numeric = val_df[numeric_cols].values.astype(np.float32)
    test_numeric = test_df[numeric_cols].values.astype(np.float32)

    # Autoencoder transform
    train_numeric_latent, val_numeric_latent, test_numeric_latent = autoencoder_transform_train_val_test(
        train_numeric, val_numeric, test_numeric,
        latent_dim=latent_dim,
        num_epochs=ae_epochs,
        ae_lr=ae_lr
    )

    train_summary = train_df["AugSummary"].values
    val_summary = val_df["AugSummary"].values
    test_summary = test_df["AugSummary"].values

    train_desc = train_df["Description"].values
    val_desc = val_df["Description"].values
    test_desc = test_df["Description"].values

    train_labels = train_df["Label"].values
    val_labels = val_df["Label"].values
    test_labels = test_df["Label"].values

    # Determine batch size
    if bs is not None:
        BATCH_SIZE = bs
    elif DYNAMIC_BATCH:
        BATCH_SIZE = int(len(train_summary) * BATCH_SIZE_RATIO)

    # Tokenize text data
    summary_train_tokens = summary_tokenization(train_summary.tolist())
    summary_val_tokens = summary_tokenization(val_summary.tolist())
    desc_train_tokens = desc_tokenization(train_desc.tolist())
    desc_val_tokens = desc_tokenization(val_desc.tolist())

    train_numeric_t = torch.tensor(train_numeric_latent, dtype=torch.float)
    val_numeric_t = torch.tensor(val_numeric_latent, dtype=torch.float)
    test_numeric_t = torch.tensor(test_numeric_latent, dtype=torch.float)

    train_y = torch.tensor(train_labels, dtype=torch.float)
    val_y = torch.tensor(val_labels, dtype=torch.float)
    test_y = torch.tensor(test_labels, dtype=torch.float)

    # Convert tokenized text to float tensors (then round later)
    train_summary_ids = torch.tensor(summary_train_tokens['input_ids'], dtype=torch.float)
    train_desc_ids = torch.tensor(desc_train_tokens['input_ids'], dtype=torch.float)
    val_summary_ids = torch.tensor(summary_val_tokens['input_ids'], dtype=torch.float)
    val_desc_ids = torch.tensor(desc_val_tokens['input_ids'], dtype=torch.float)

    # Build final input: [autoenc. numeric (latent_dim) | summary tokens (SEQ_LEN) | desc tokens (SEQ_LEN)]
    train_seq = torch.cat([train_numeric_t, train_summary_ids, train_desc_ids], dim=1)
    val_seq = torch.cat([val_numeric_t, val_summary_ids, val_desc_ids], dim=1)

    # Dataloaders
    train_dataloader = prepare_dataloader(train_seq, train_y, sampler_type='random')
    val_dataloader = prepare_dataloader(val_seq, val_y, sampler_type='sequential')

    # Test
    test_summary_tokens = summary_tokenization(test_summary.tolist())
    test_desc_tokens = desc_tokenization(test_desc.tolist())
    test_summary_ids = torch.tensor(test_summary_tokens['input_ids'], dtype=torch.float)
    test_desc_ids = torch.tensor(test_desc_tokens['input_ids'], dtype=torch.float)
    test_seq = torch.cat([test_numeric_t, test_summary_ids, test_desc_ids], dim=1)
    test_dataloader = prepare_dataloader(test_seq, test_y, sampler_type='sequential')

    return file_pair, train_dataloader, val_dataloader, [test_dataloader], [file_pair]


def prepare_dataframe(file_name):
    df = pd.read_csv(file_name).fillna("")
    df.rename(columns={'Custom field (Story Points)': 'Label'}, inplace=True)
    return df


def data_split(df):
    """
    Within-project split: 60% train, 20% val, 20% test
    """
    print("within project split!")
    n = len(df)
    train_val_split = int(n * 0.6)
    val_test_split = int(n * 0.8)
    train_df = df.iloc[:train_val_split].copy()
    val_df = df.iloc[train_val_split:val_test_split].copy()
    test_df = df.iloc[val_test_split:].copy()
    return train_df, val_df, test_df


###############################################################################
# 2) TOKENIZATION: SUMMARY (BERT) & DESCRIPTION (CodeBERT)
###############################################################################
def summary_tokenization(text_list):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    return tokenizer(
        text_list,
        truncation=True,
        max_length=SEQUENCE_LEN,
        padding='max_length'
    )


def desc_tokenization(text_list):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    return tokenizer(
        text_list,
        truncation=True,
        max_length=SEQUENCE_LEN,
        padding='max_length'
    )


def prepare_dataloader(seq, y, sampler_type):
    global BATCH_SIZE
    ds = TensorDataset(seq, y)
    sampler = RandomSampler(ds) if sampler_type == 'random' else SequentialSampler(ds)
    return DataLoader(ds, sampler=sampler, batch_size=BATCH_SIZE)


###############################################################################
# 3) MODEL DEFINITION: Dual LLM (BERT + CodeBERT)
###############################################################################
class DualLLMForSequence(nn.Module):
    def __init__(self, pca_dims=5):
        super().__init__()
        # Summaries => BERT
        self.bert = AutoModel.from_pretrained("bert-base-cased")
        for param in self.bert.parameters():
            param.requires_grad = False

        # Descriptions => CodeBERT
        self.codebert = AutoModel.from_pretrained("microsoft/codebert-base")
        for param in self.codebert.parameters():
            param.requires_grad = False

        self.pca_dims = pca_dims  # latent dimension from the autoencoder
        self.summary_proj = nn.Linear(768, 3)
        self.desc_proj = nn.Linear(768, 3)

        # numeric (pca_dims) + summary(3) + description(3) = pca_dims + 6
        in_dim = pca_dims + 6
        self.hidden2 = nn.Linear(in_dim, 50)
        self.score = nn.Linear(50, 1)

    def forward(self, x):
        # x shape: [batch, pca_dims + SEQUENCE_LEN + SEQUENCE_LEN]
        numeric_feats = x[:, :self.pca_dims]

        # For token IDs, round float values and cast to long
        summary_ids = x[:, self.pca_dims: self.pca_dims + SEQUENCE_LEN].round().long()
        desc_ids = x[:, self.pca_dims + SEQUENCE_LEN: self.pca_dims + 2 * SEQUENCE_LEN].round().long()

        # BERT
        sum_out = self.bert(summary_ids)
        sum_cls = sum_out.last_hidden_state[:, 0, :]
        sum_3d = self.summary_proj(sum_cls)

        # CodeBERT
        desc_out = self.codebert(desc_ids)
        desc_cls = desc_out.last_hidden_state[:, 0, :]
        desc_3d = self.desc_proj(desc_cls)

        combined = torch.cat([numeric_feats, sum_3d, desc_3d], dim=1)
        combined = torch.relu(self.hidden2(combined))
        return self.score(combined)


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
    # -------- 1. Suggest hyperparameters --------
    latent_dim = trial.suggest_categorical("latent_dim", [3, 5, 10])
    ae_lr = trial.suggest_loguniform("ae_lr", 1e-5, 1e-3)
    model_lr = trial.suggest_loguniform("model_lr", 1e-5, 1e-3)
    ae_epochs = trial.suggest_categorical("ae_epochs", [20, 50])
    bs = trial.suggest_categorical("batch_size", [8, 16, 32])

    # -------- 2. Build data loaders --------
    # data_processing signature:
    # data_processing(file_pair, latent_dim, ae_epochs, ae_lr, lr, bs)
    file_pair_out, train_dl, val_dl, test_dls, test_file_names = data_processing(
        file_pair=file_pair,
        latent_dim=latent_dim,
        ae_epochs=ae_epochs,
        ae_lr=ae_lr,
        bs=bs
    )

    # -------- 3. Instantiate the model --------
    model = DualLLMForSequence(pca_dims=latent_dim).to(DEVICE)

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
        MODEL = DualLLMForSequence().to(DEVICE)
        file_pair, train_dl, val_dl, test_dls, test_file_names = data_processing(file_pair=file)
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
    result_path = 'results/hyper_param_tuning/autoencoders/'
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

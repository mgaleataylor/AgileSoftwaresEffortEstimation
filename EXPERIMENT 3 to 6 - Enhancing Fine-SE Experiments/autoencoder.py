import pandas as pd
import os
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import numpy as np
import time
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
# GLOBAL CONFIG
###############################################################################

EPOCHS = 20
BATCH_SIZE_RATIO = 0.3
SEQUENCE_LEN = 20
LEARNING_RATE = 5e-4
OUTPUT = ''
MODEL = None
DYNAMIC_BATCH = True
# BATCH_SIZE = 16
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


def autoencoder_transform_train_val_test(x_train, x_val, x_test, latent_dim=5, num_epochs=50, ae_lr=3e-4,
                                         save_path="data/AUTOENCODER_MODEL"):
    input_dim = x_train.shape[1]
    autoencoder = NumericAutoencoder(input_dim, latent_dim).to(DEVICE)

    # if os.path.exists(save_path):
    #    print("🔄 Loading pre-trained autoencoder weights...")
    #    autoencoder.load_state_dict(torch.load(save_path + '/autoencoder.pth'))
    # else:
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=ae_lr)
    loss_fn = nn.MSELoss()
    x_train_tensor = torch.tensor(x_train, dtype=torch.float).to(DEVICE)
    dataset = TensorDataset(x_train_tensor)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    for epoch in range(num_epochs):
        autoencoder.train()
        for batch in dataloader:
            batch = batch[0]
            optimizer.zero_grad()
            reconstruction, _ = autoencoder(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()

        # if not os.path.exists(save_path):
        #    os.makedirs(save_path)

        # torch.save(autoencoder.state_dict(), save_path + '/autoencoder.pth')

    autoencoder.eval()
    with torch.no_grad():
        X_train_latent = autoencoder.encoder(torch.tensor(x_train, dtype=torch.float).to(DEVICE)).cpu().numpy()
        X_val_latent = autoencoder.encoder(torch.tensor(x_val, dtype=torch.float).to(DEVICE)).cpu().numpy()
        X_test_latent = autoencoder.encoder(torch.tensor(x_test, dtype=torch.float).to(DEVICE)).cpu().numpy()

    return X_train_latent, X_val_latent, X_test_latent


###############################################################################
# 1) DATA PROCESSING / LOADING & Autoencoder for Numeric Features
###############################################################################
def data_processing(file_pair, latent_dim, ae_epochs, ae_lr, bs=None):
    global BATCH_SIZE, BATCH_SIZE_RATIO, DATA_PATH

    fname = DATA_PATH + file_pair
    df = prepare_dataframe(fname)

    df["AugSummary"] = (
            df["Issue Type"].astype(str) + " " +
            df["Version"].astype(str) + " " +
            df["Summary"].astype(str)
    )

    train_df, val_df, test_df = data_split(df)

    excluded = ["AugSummary", "Description", "Label", "Issue Type", "Version", "Summary"]
    numeric_cols = [col for col in df.columns if col not in excluded]

    train_numeric = train_df[numeric_cols].values.astype(np.float32)
    val_numeric = val_df[numeric_cols].values.astype(np.float32)
    test_numeric = test_df[numeric_cols].values.astype(np.float32)

    train_numeric_latent, val_numeric_latent, test_numeric_latent = autoencoder_transform_train_val_test(
        train_numeric, val_numeric, test_numeric, latent_dim=latent_dim, num_epochs=ae_epochs, ae_lr=ae_lr
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

    if bs is not None:
        BATCH_SIZE = bs
    elif DYNAMIC_BATCH:
        BATCH_SIZE = int(len(train_summary) * BATCH_SIZE_RATIO)

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

    # Convert tokenized text to tensors as floats to preserve exact integer values
    train_summary_ids = torch.tensor(summary_train_tokens['input_ids'], dtype=torch.float)
    train_desc_ids = torch.tensor(desc_train_tokens['input_ids'], dtype=torch.float)
    val_summary_ids = torch.tensor(summary_val_tokens['input_ids'], dtype=torch.float)
    val_desc_ids = torch.tensor(desc_val_tokens['input_ids'], dtype=torch.float)

    # Build final input: [Autoencoder numeric (latent_dim) | summary tokens (SEQ_LEN) | desc tokens (SEQ_LEN)]
    train_seq = torch.cat([train_numeric_t, train_summary_ids, train_desc_ids], dim=1)
    train_dataloader = prepare_dataloader(train_seq, train_y, sampler_type='random', bs=BATCH_SIZE)

    val_seq = torch.cat([val_numeric_t, val_summary_ids, val_desc_ids], dim=1)
    val_dataloader = prepare_dataloader(val_seq, val_y, sampler_type='sequential', bs=BATCH_SIZE)

    test_summary_tokens = summary_tokenization(test_summary.tolist())
    test_desc_tokens = desc_tokenization(test_desc.tolist())
    test_summary_ids = torch.tensor(test_summary_tokens['input_ids'], dtype=torch.float)
    test_desc_ids = torch.tensor(test_desc_tokens['input_ids'], dtype=torch.float)
    test_seq = torch.cat([test_numeric_t, test_summary_ids, test_desc_ids], dim=1)
    test_dataloader = prepare_dataloader(test_seq, test_y, sampler_type='sequential', bs=BATCH_SIZE)

    return file_pair, train_dataloader, val_dataloader, [test_dataloader], [file_pair]


def prepare_dataframe(file_name):
    df = pd.read_csv(file_name).fillna("")
    df.rename(columns={'Custom field (Story Points)': 'Label'}, inplace=True)
    return df


def data_split(df):
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


def prepare_dataloader(seq, y, sampler_type, bs=None):
    global BATCH_SIZE

    if bs is not None:
        batch_size = bs
    else:
        batch_size = BATCH_SIZE

    ds = TensorDataset(seq, y)
    sampler = RandomSampler(ds) if sampler_type == 'random' else SequentialSampler(ds)
    return DataLoader(ds, sampler=sampler, batch_size=batch_size)


###############################################################################
# 3) MODEL DEFINITION: Dual LLM (LoRA on BERT) + Autoencoder for Numeric Features
###############################################################################
# class DualLLMForSequence(nn.Module):
#     def __init__(self, pca_dims=5):
#         super().__init__()
#         # Summaries => BERT
#         self.bert = AutoModel.from_pretrained("bert-base-cased")
#         for param in self.bert.parameters():
#             param.requires_grad = False
#
#         # Descriptions => CodeBERT
#         self.codebert = AutoModel.from_pretrained("microsoft/codebert-base")
#         for param in self.codebert.parameters():
#             param.requires_grad = False
#
#         self.pca_dims = pca_dims  # latent dimension from the autoencoder
#         self.summary_proj = nn.Linear(768, 3)
#         self.desc_proj = nn.Linear(768, 3)
#         in_dim = pca_dims + 6  # numeric (pca_dims) + summary (3) + description (3)
#         self.hidden2 = nn.Linear(in_dim, 50)
#         self.score = nn.Linear(50, 1)
#
#     def forward(self, x):
#         # x shape: [batch, pca_dims + SEQUENCE_LEN + SEQUENCE_LEN]
#         numeric_feats = x[:, :self.pca_dims]
#         # For token IDs, first round the float values and convert to long
#         summary_ids = x[:, self.pca_dims: self.pca_dims + SEQUENCE_LEN].round().long()
#         desc_ids = x[:, self.pca_dims + SEQUENCE_LEN: self.pca_dims + 2 * SEQUENCE_LEN].round().long()
#
#         sum_out = self.bert(summary_ids)
#         sum_cls = sum_out.last_hidden_state[:, 0, :]
#         sum_3d = self.summary_proj(sum_cls)
#
#         desc_out = self.codebert(desc_ids)
#         desc_cls = desc_out.last_hidden_state[:, 0, :]
#         desc_3d = self.desc_proj(desc_cls)
#
#         combined = torch.cat([numeric_feats, sum_3d, desc_3d], dim=1)
#         combined = torch.relu(self.hidden2(combined))
#         return self.score(combined)

class EnsembleLLMWithAutoencoder(nn.Module):
    def __init__(self, pca_dims: int = 5):
        super().__init__()
        self.pca_dims = pca_dims

        # 1) BERT for summary
        self.bert = AutoModel.from_pretrained("bert-base-cased")
        for name, param in self.bert.named_parameters():
            # only fine‑tune the last two layers
            if name.startswith("encoder.layer.10") or name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False

        # 2) CodeBERT for description
        self.codebert = AutoModel.from_pretrained("microsoft/codebert-base")
        for name, param in self.codebert.named_parameters():
            if name.startswith("encoder.layer.10") or name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False

        # project [CLS] embeddings from 768 → 3 dim
        self.summary_proj = nn.Linear(768, 3)
        self.desc_proj = nn.Linear(768, 3)

        # MLP: numeric (pca_dims) + 3 + 3 → 50 → 1
        in_dim = pca_dims + 3 + 3
        self.hidden = nn.Linear(in_dim, 50)
        self.score = nn.Linear(50, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, pca_dims + SEQ_LEN + SEQ_LEN]
          - x[:, :pca_dims]                     → numeric features
          - x[:, pca_dims : pca_dims+SEQ_LEN]   → summary input_ids
          - x[:, pca_dims+SEQ_LEN : ]           → desc input_ids
        """
        # 1) numeric
        numeric_feats = x[:, :self.pca_dims].float()

        # 2) summary tokens + auto‐mask
        sum_ids = x[:, self.pca_dims: self.pca_dims + SEQUENCE_LEN].long()
        sum_mask = (sum_ids != 0).long()
        sum_out = self.bert(input_ids=sum_ids, attention_mask=sum_mask)
        sum_cls = sum_out.last_hidden_state[:, 0, :]  # [batch,768]
        sum_3d = self.summary_proj(sum_cls)  # [batch,  3]

        # 3) description tokens + auto‐mask
        desc_ids = x[:, self.pca_dims + SEQUENCE_LEN:].long()
        desc_mask = (desc_ids != 0).long()
        desc_out = self.codebert(input_ids=desc_ids, attention_mask=desc_mask)
        desc_cls = desc_out.last_hidden_state[:, 0, :]
        desc_3d = self.desc_proj(desc_cls)

        # 4) combine & score
        combined = torch.cat([numeric_feats, sum_3d, desc_3d], dim=1)
        hid = torch.relu(self.hidden(combined))
        return self.score(hid)  # [batch,1]


###############################################################################
# 4) TRAIN/EVAL/TEST
###############################################################################
def train_eval_test(file_pair, train_dl, val_dl, test_dl, model, lr):
    global EPOCHS, MAE_RECORDS, MDAE_RECORDS, DEVICE

    if lr is not None:
        lr_rate = lr
    else:
        lr_rate = LEARNING_RATE

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate)
    total_steps = int(len(train_dl)) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)
    print(f"Start training for {file_pair} with lr={lr_rate}, epochs={EPOCHS}...")
    min_eval_loss = float('inf')
    best_epoch = 0
    time_records = []
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
        if avg_eval_loss < min_eval_loss:
            min_eval_loss = avg_eval_loss
            best_epoch = e
        print("===============================")
        # TEST
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
            # calculate errors
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
    return min_eval_loss


def define_params_optuna(file):
    if 'mesos' in file.lower():
        latent_dim = 5
        ae_lr = 9.304175776828763e-05
        lr = 0.000567758902789303
        ae_epochs = 50
        batch_size = 32
    if 'usergrid' in file.lower():
        latent_dim = 5
        ae_lr = 0.0009985706980278428
        lr = 0.0009567813417645984
        ae_epochs = 20
        batch_size = 8
    if 'data_management' in file.lower():
        latent_dim = 10
        ae_lr = 0.00045973705603571474
        lr = 0.0008784969754294248
        ae_epochs = 50
        batch_size = 8

    return latent_dim, ae_lr, lr, ae_epochs, batch_size


def define_params_grid(file):
    if 'mesos' in file.lower():
        latent_dim = 5
        ae_lr = 1e-05
        lr = 0.001
        ae_epochs = 20
        batch_size = 32
    if 'usergrid' in file.lower():
        latent_dim = 5
        ae_lr = 0.0001
        lr = 0.001
        ae_epochs = 50
        batch_size = 8
    if 'data_management' in file.lower():
        latent_dim = 10
        ae_lr = 0.00045973705603571474
        lr = 0.0008784969754294248
        ae_epochs = 50
        batch_size = 8

    return latent_dim, ae_lr, lr, ae_epochs, batch_size


def main(result_path, files, optuna, hyper_params):
    global MODEL

    for file in files:
        latent_dim, ae_lr, lr, ae_epochs, batch_size = (
            (define_params_optuna(file) if optuna else define_params_grid(file))
            if hyper_params
            else (5, 3e-4, 5e-4, 50, None) # Setting latent pc dimension 5 for default if no hyper param
        )

        MODEL = EnsembleLLMWithAutoencoder(pca_dims=latent_dim).to(DEVICE)
        file_pair, train_dl, val_dl, test_dls, test_file_names = data_processing(file_pair=file, latent_dim=latent_dim,
                                                                                 ae_lr=ae_lr, ae_epochs=ae_epochs,
                                                                                 bs=batch_size)
        train_eval_test(file_pair, train_dl, val_dl, test_dls, MODEL, lr=lr)
        del MODEL
        torch.cuda.empty_cache()
        global OUTPUT
        file_result_path = result_path + file[:-4] + '.txt'
        os.makedirs(os.path.dirname(file_result_path), exist_ok=True)
        with open(file_result_path, 'w+') as f:
            f.writelines(OUTPUT)
        OUTPUT = ""


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

    optuna = False
    hyper_param = False

    if hyper_param:
        main(result_path='results/autoencoders/hyper_param/', files=train_test_file_paths, optuna=optuna,
             hyper_params=hyper_param)
    else:
        main(result_path='results/autoencoders/default/', files=train_test_file_paths, optuna=optuna,
             hyper_params=hyper_param)

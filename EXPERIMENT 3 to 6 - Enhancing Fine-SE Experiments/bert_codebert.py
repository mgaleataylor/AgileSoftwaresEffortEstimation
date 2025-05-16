import openpyxl
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import numpy as np
import time
import os
import torch.nn as nn

# GLOBAL CONFIG
projectnum = 2

EPOCHS = 20
BATCH_SIZE_RATIO = 0.3
SEQUENCE_LEN = 20
LEARNING_RATE = 5e-4
DROPOUT_PROB = 0.1
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


###############################################################################
# 1) DATA PROCESSING
###############################################################################
def data_processing(file_pair, bs=None):
    global BATCH_SIZE, BATCH_SIZE_RATIO, DATA_PATH, WITHIN_PROJECT, DYNAMIC_BATCH

    fname = DATA_PATH + file_pair
    df = prepare_dataframe(fname)

    df["Summary"] = (
            df["Issue Type"].astype(str) + " " +
            df["Version"].astype(str) + " " +
            df["Summary"].astype(str)
    )

    (
        train_ex, train_summary, train_desc, train_labels,
        val_ex, val_summary, val_desc, val_labels,
        test_ex, test_summary, test_desc, test_labels
    ) = data_split(df)

    if bs is not None:
        BATCH_SIZE = bs
    elif DYNAMIC_BATCH:
        BATCH_SIZE = int(len(train_summary) * BATCH_SIZE_RATIO)

    # Tokenize summary and description separately
    summary_train_tokens = summary_tokenization(train_summary.tolist())
    summary_val_tokens = summary_tokenization(val_summary.tolist())

    desc_train_tokens = desc_tokenization(train_desc.tolist())
    desc_val_tokens = desc_tokenization(val_desc.tolist())

    # Convert numeric features to tensors
    train_ex_t = torch.tensor(np.array(train_ex))
    train_y = torch.tensor(train_labels.tolist()).float()

    val_ex_t = torch.tensor(np.array(val_ex))
    val_y = torch.tensor(val_labels.tolist()).float()

    # Prepare sequences
    train_summary_ids = torch.tensor(summary_train_tokens['input_ids'])
    train_desc_ids = torch.tensor(desc_train_tokens['input_ids'])
    train_seq = torch.cat([train_ex_t, train_summary_ids, train_desc_ids], dim=1)

    val_summary_ids = torch.tensor(summary_val_tokens['input_ids'])
    val_desc_ids = torch.tensor(desc_val_tokens['input_ids'])
    val_seq = torch.cat([val_ex_t, val_summary_ids, val_desc_ids], dim=1)

    train_dataloader = prepare_dataloader(train_seq, train_y, sampler_type='random', bs=BATCH_SIZE)
    val_dataloader = prepare_dataloader(val_seq, val_y, sampler_type='sequential', bs=BATCH_SIZE)

    # Prepare testing data
    test_ex_t = torch.tensor(np.array(test_ex))
    test_y = torch.tensor(test_labels.tolist()).float()

    test_summary_tokens = summary_tokenization(test_summary.tolist())
    test_desc_tokens = desc_tokenization(test_desc.tolist())

    test_summary_ids = torch.tensor(test_summary_tokens['input_ids'])
    test_desc_ids = torch.tensor(test_desc_tokens['input_ids'])
    test_seq = torch.cat([test_ex_t, test_summary_ids, test_desc_ids], dim=1)

    test_dataloader = prepare_dataloader(test_seq, test_y, sampler_type='sequential', bs=BATCH_SIZE)

    return file_pair, train_dataloader, val_dataloader, [test_dataloader]


def prepare_dataframe(file_name):
    """
    Assumes CSV has: [Assignee_count, Reporter_count, Creator_count, Summary, Description, Custom field (Story Points)]
    We'll keep them separate (not combining them) so we can do separate tokenization for summary & description.
    """
    order = [
        'Contributors', 'Developer_Rank', 'Developer_commits', 'Developer_created_MRs', 'Developer_modified_files',
        'Developer_commits_reviews', 'Developer_updated_MRs', 'Developer_fixed_defects', 'Developer_ARs',
        'Tester_detected_defects', 'Tester_ARs', 'Creator_ARs', 'Version_encoded',
        'Issue Type',
        'Version',
        'Summary',
        'Description',
        'Custom field (Story Points)'
    ]
    data = pd.read_csv(file_name)
    data = data[order].fillna("")
    data.rename(columns={'Custom field (Story Points)': 'Label'}, inplace=True)
    return data


def data_split(df):
    """
    Return numeric_ex, summary_text, desc_text, label for train/val/test splits.
    """
    train_val_split = int(len(df) * 0.6)
    val_test_split = int(len(df) * 0.8)

    # TRAIN
    train_ex = df.iloc[:train_val_split, 0:13]
    train_summary = df['Summary'][:train_val_split]
    train_desc = df['Description'][:train_val_split]
    train_labels = df['Label'][:train_val_split]

    # VAL
    val_ex = df.iloc[train_val_split:val_test_split, 0:13]
    val_summary = df['Summary'][train_val_split:val_test_split]
    val_desc = df['Description'][train_val_split:val_test_split]
    val_labels = df['Label'][train_val_split:val_test_split]

    # TEST
    test_ex = df.iloc[val_test_split:, 0:13]
    test_summary = df['Summary'][val_test_split:]
    test_desc = df['Description'][val_test_split:]
    test_labels = df['Label'][val_test_split:]

    return (
        train_ex, train_summary, train_desc, train_labels,
        val_ex, val_summary, val_desc, val_labels,
        test_ex, test_summary, test_desc, test_labels
    )


##############################################################################
# 2) TOKENIZATION FOR SUMMARY & DESCRIPTION (Separate)
##############################################################################
def summary_tokenization(text_list):
    """
    Use a normal language model tokenizer for summary,
    e.g. 'bert-base-cased' or 'roberta-base'.
    """
    summary_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    return summary_tokenizer(
        text_list,
        truncation=True,
        max_length=SEQUENCE_LEN,
        padding='max_length'
    )


def desc_tokenization(text_list):
    """
    Use CodeBERT for the description, which may contain code.
    """
    desc_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    return desc_tokenizer(
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


class EnsembleLLMForSequence(nn.Module):
    """
    Ensemble of BERT (for NL features) and CodeBERT (for code-rich descriptions),
    concatenated with numeric expert features.
    """

    def __init__(self, sequence_len=20, numeric_feats=13, activation_fn=nn.ReLU(), dropout_prob=0.1, hidden_dim=50):
        super(EnsembleLLMForSequence, self).__init__()
        self.sequence_len = sequence_len
        self.numeric_feats = numeric_feats
        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout_prob)

        # Add normalization for numeric features
        self.numeric_norm = nn.LayerNorm(numeric_feats)

        # 1) BERT for general textual fields (Summary, Issue Type, Version)
        self.bert = AutoModel.from_pretrained("bert-base-cased")
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True

        # Add normalization for BERT outputs
        self.bert_norm = nn.LayerNorm(768)

        # 2) CodeBERT for descriptions containing code snippets
        self.codebert = AutoModel.from_pretrained("microsoft/codebert-base")
        for param in self.codebert.parameters():
            param.requires_grad = False
        for param in self.codebert.encoder.layer[-2:].parameters():
            param.requires_grad = True

        # Add normalization for CodeBERT outputs
        self.codebert_norm = nn.LayerNorm(768)

        # Improved projection layers with normalization
        self.summary_proj = nn.Sequential(
            nn.Linear(768, 3),
            nn.LayerNorm(3),
            self.activation_fn,
            self.dropout
        )

        self.desc_proj = nn.Sequential(
            nn.Linear(768, 3),
            nn.LayerNorm(3),
            self.activation_fn,
            self.dropout
        )

        # Improved fusion layers
        fused_dim = numeric_feats + 6  # numeric + summary(3) + desc(3)
        self.hidden_layer = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            self.activation_fn,
            self.dropout
        )
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids):
        # Process numeric features
        numeric = input_ids[:, :self.numeric_feats].float()
        numeric = self.numeric_norm(numeric)  # Added normalization

        # Process summary with BERT
        summary_ids = input_ids[:, self.numeric_feats:self.numeric_feats + self.sequence_len].long()
        summary_outputs = self.bert(input_ids=summary_ids)
        summary_cls = summary_outputs.last_hidden_state[:, 0, :]
        summary_cls = self.bert_norm(summary_cls)  # Added normalization
        summary_emb = self.summary_proj(summary_cls)

        # Process description with CodeBERT
        desc_ids = input_ids[:, self.numeric_feats + self.sequence_len:].long()
        desc_outputs = self.codebert(input_ids=desc_ids)
        desc_cls = desc_outputs.last_hidden_state[:, 0, :]
        desc_cls = self.codebert_norm(desc_cls)  # Added normalization
        desc_emb = self.desc_proj(desc_cls)

        # Fusion and final processing
        fused = torch.cat([numeric, summary_emb, desc_emb], dim=1)
        hidden = self.hidden_layer(fused)
        output = self.output_layer(hidden)
        return output


##############################################################################
# 4) TRAIN/EVAL/TEST
##############################################################################
def train_eval_test(file_pair, train_dl, val_dl, test_dls, model, lr):
    global LEARNING_RATE, EPOCHS, MAE_RECORDS, MDAE_RECORDS, DEVICE

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
        model_lr = 0.00016510191183686046
        batch_size = 8
        activation_fn = nn.ReLU()
        dropout_prob = 0.17254127061179908
        hidden_dim = 32
    if 'usergrid' in file.lower():
        model_lr = 0.0005893794280121997
        batch_size = 8
        activation_fn = nn.LeakyReLU()
        dropout_prob = 0.46861628853452364
        hidden_dim = 32
    if 'data_management' in file.lower():
        model_lr = 9.76439437396477e-05
        batch_size = 8
        activation_fn = nn.Tanh()
        dropout_prob = 0.4715909530565474
        hidden_dim = 64

    return model_lr, batch_size, activation_fn, dropout_prob, hidden_dim


def main(result_path, files, hyper_params):
    global MODEL

    for file in files:
        model_lr, batch_size, activation_fn, dropout_prob, hidden_dim = (
            (define_params_optuna(file))
            if hyper_params
            else (LEARNING_RATE, None, nn.ReLU(), 0.1, 50)
        )

        MODEL = EnsembleLLMForSequence().to(DEVICE)  # Our new dual LLM model
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

        global projectnum
        projectnum += 1


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

    optuna = False
    hyper_param = False

    if hyper_param:
        main(result_path='results/bert_codebert/hyper_param/', files=train_test_file_paths, hyper_params=hyper_param)
    else:
        main(result_path='results/bert_codebert/default/', files=train_test_file_paths, hyper_params=hyper_param)

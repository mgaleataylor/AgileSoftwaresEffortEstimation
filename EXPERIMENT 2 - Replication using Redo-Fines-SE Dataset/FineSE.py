import openpyxl
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel, BertPreTrainedModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import numpy as np
import time
import os
import torch.nn as nn

global EPOCHS, BATCH_SIZE_RATIO, SEQUENCE_LEN, LEARNING_RATE, TOKENIZER, MODEL_NAME

projectnum = 2
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
global DEVICE, DATA_PATH
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
ROW_MAE, ROW_MMRE, ROW_PRED = 3, 4, 5


def data_processing(file_pair):
    global BATCH_SIZE, BATCH_SIZE_RATIO, DATA_PATH, WITHIN_PROJECT, DYNAMIC_BATCH

    train_data = pd.DataFrame()
    fname = DATA_PATH + file_pair
    df = prepare_dataframe(fname)
    train_data = df

    # data split
    if WITHIN_PROJECT:
        train_ex, train_text, train_labels, val_ex, val_text, val_labels, test_ex, test_text, test_labels = within_project_split(
            train_data)
    # define batch size dynamicalloutputsy based on training length
    if DYNAMIC_BATCH:
        BATCH_SIZE = int(len(train_text) * BATCH_SIZE_RATIO)
    # tokenization
    tokens_train = tokenization(train_text.tolist())
    tokens_val = tokenization(val_text.tolist())

    train_seq = torch.tensor(tokens_train['input_ids'])
    train_ex = np.array(train_ex)
    train_ex = torch.tensor(train_ex)
    train_y = torch.tensor(train_labels.tolist()).type(torch.FloatTensor)
    train_seq = torch.cat((train_ex, train_seq), dim=1)
    train_dataloader = prepare_dataloader(train_seq, train_y, sampler_type='random')

    val_seq = torch.tensor(tokens_val['input_ids'])
    val_ex = np.array(val_ex)
    val_ex = torch.tensor(val_ex)
    val_y = torch.tensor(val_labels.tolist()).type(torch.FloatTensor)
    val_seq = torch.cat((val_ex, val_seq), dim=1)
    val_dataloader = prepare_dataloader(val_seq, val_y, sampler_type='sequential')

    # prepare testing datasets
    all_test_dataloader = []
    test_file_names = []
    if WITHIN_PROJECT:
        tokens_test = tokenization(test_text.tolist())
        test_seq = torch.tensor(tokens_test['input_ids'])
        test_ex = np.array(test_ex)
        test_ex = torch.tensor(test_ex)
        test_seq = torch.cat((test_ex, test_seq), dim=1)
        test_y = torch.tensor(test_labels.tolist()).type(torch.FloatTensor)
        test_dataloader = prepare_dataloader(test_seq, test_y, sampler_type='sequential')
        all_test_dataloader.append(test_dataloader)
        test_file_names.append(file_pair)
        return file_pair, train_dataloader, val_dataloader, all_test_dataloader, test_file_names


def tokenization(text_list):
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    return tokenizer(text_list, truncation=True, max_length=SEQUENCE_LEN, padding='max_length')


def prepare_dataframe(file_name):
    data = pd.read_csv(file_name)
    order = ['Assignee_count', 'Reporter_count', 'Creator_count', 'Summary', 'Custom field (Story Points)']
    data = data[order]
    data = data.fillna(0)
    return pd.DataFrame(data=data)


def prepare_dataloader(seq, y, sampler_type):
    global BATCH_SIZE
    tensor_dataset = TensorDataset(seq, y)
    if sampler_type == 'random':
        sampler = RandomSampler(tensor_dataset)
    elif sampler_type == 'sequential':
        sampler = SequentialSampler(tensor_dataset)
    dataloader = DataLoader(tensor_dataset, sampler=sampler, batch_size=BATCH_SIZE)
    return dataloader


def within_project_split(data):
    print('within project split!')
    train_val_split_point = int(len(data) * 0.6)
    val_test_split_point = int(len(data) * 0.8)
    train_ex = data.iloc[:train_val_split_point, 0:3]
    train_text = data['Summary'][:train_val_split_point]
    train_labels = (data['Custom field (Story Points)'][:train_val_split_point])
    val_ex = data.iloc[train_val_split_point:val_test_split_point, 0:3]
    val_text = data['Summary'][train_val_split_point:val_test_split_point]
    val_labels = (data['Custom field (Story Points)'][train_val_split_point:val_test_split_point])
    test_ex = data.iloc[val_test_split_point:, 0:3]
    test_text = data['Summary'][val_test_split_point:]
    test_labels = (data['Custom field (Story Points)'][val_test_split_point:])
    return train_ex, train_text, train_labels, val_ex, val_text, val_labels, test_ex, test_text, test_labels


class BertForSequence(nn.Module):
    def __init__(self):
        super(BertForSequence, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
        self.hidden1 = nn.Linear(768, 3)
        self.hidden2 = nn.Linear(6, 50)
        self.score = nn.Linear(50, 1)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs_bert = self.bert(input_ids[:, 3:].long(), token_type_ids, attention_mask)
        outputs = outputs_bert.last_hidden_state[:, 0, :]
        outputs = self.hidden1(outputs)
        outputs = torch.cat((input_ids[:, 0:3], outputs), dim=1)
        outputs = torch.relu(self.hidden2(outputs.float()))
        logit = self.score(outputs)
        return logit


def train_eval_test(file_pair, train_dataloader, val_dataloader, all_test_dataloader, model, test_file_names):
    global LEARNING_RATE, EPOCHS, MAE_RECORDS, MDAE_RECORDS, DEVICE
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    print("Start training for ", file_pair, ".....")

    min_eval_loss_epoch = [float('inf'), 0]
    early_stop_counter = 0
    max_no_improvement = 3  # Stop training if no improvement for these many epochs

    time_records = []
    MAE_RECORDS, MDAE_RECORDS, MMRE_RECORDS, PRED_RECORDS = [], [], [], []
    start_time = time.time()
    loss_fct = nn.L1Loss()

    for e in range(EPOCHS):
        if early_stop_counter >= max_no_improvement:
            print("No improvement for {} epochs, stopping training early.".format(max_no_improvement))
            break

        torch.cuda.empty_cache()
        print(">>> Epoch ", e)

        # ---TRAINING---
        model.train()
        total_train_loss = 0
        for step, batch in enumerate(train_dataloader):
            b_input_ids, b_labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
            model.zero_grad()
            result = model(b_input_ids, labels=b_labels)
            loss = loss_fct(result, b_labels)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            del step, batch, b_input_ids, b_labels, result, loss

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(" Average training MAE loss: {0:.2f}".format(avg_train_loss))
        del avg_train_loss, total_train_loss

        time_records.append(time.time() - start_time)

        # ---EVALUATION---
        print("-")
        model.eval()
        total_eval_loss = 0
        for batch in val_dataloader:
            b_input_ids = batch[0].to(DEVICE)
            b_labels = batch[1].to(DEVICE)  # Reshape labels to match model output
            with torch.no_grad():
                result = model(b_input_ids)
                loss = loss_fct(result, b_labels)
                total_eval_loss += loss.item()
                del batch, b_input_ids, b_labels, result, loss

        avg_eval_loss = total_eval_loss / len(val_dataloader)
        print(" Average eval MAE loss: {0:.2f}".format(avg_eval_loss))

        if avg_eval_loss < min_eval_loss_epoch[0]:
            min_eval_loss_epoch = [avg_eval_loss, e]
            early_stop_counter = 0  # Reset counter if improvement is found
        else:
            early_stop_counter += 1

        del avg_eval_loss, total_eval_loss
        print("===============================")

        # ---TESTING---
        for idx, test_dataloader in enumerate(all_test_dataloader):
            test_file_name = test_file_names[idx]
            predictions, true_labels = [], []
            for batch in test_dataloader:
                b_input_ids, b_labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
                with torch.no_grad():
                    logits = model(b_input_ids)
                predictions.append(logits.detach().cpu().numpy())
                true_labels.append(b_labels.cpu().numpy())
                del batch, b_input_ids, b_labels, logits

            total_distance, total_mre, correct_mre_count = 0, 0, 0
            distance_records, total_data_points = [], 0

            for pred, label in zip(predictions, true_labels):
                for p, l in zip(pred, label):
                    distance = abs(p - l)
                    mre = (distance / l) if l > 0 else (distance + 1) / (l + 1)
                    if mre < 0.5:
                        correct_mre_count += 1
                    total_distance += distance
                    total_mre += mre
                    distance_records.append(distance)
                    total_data_points += 1

            MAE = total_distance / total_data_points
            MMRE = total_mre / total_data_points
            MdAE = np.median(np.array(distance_records))
            PRED = correct_mre_count / total_data_points

            MAE_RECORDS.append(MAE)
            MDAE_RECORDS.append(MdAE)
            MMRE_RECORDS.append(MMRE)
            PRED_RECORDS.append(PRED)

            print(f"Test File: {test_file_name}")
            print(" MAE: ", MAE)
            print(" MdAE: ", MdAE)
            print(" MMRE: ", MMRE)
            print(" PRED: ", PRED)

            global OUTPUT
            OUTPUT += 'Epochs ' + str(e) + '\n'
            OUTPUT += 'MAE: ' + str(MAE) + '\n'
            OUTPUT += 'MdAE: ' + str(MdAE) + '\n'
            OUTPUT += 'MMRE: ' + str(MMRE) + '\n'
            OUTPUT += 'PRED: ' + str(PRED) + '\n\n'

    OUTPUT += str(MAE_RECORDS[min_eval_loss_epoch[1]]) + '\n' + str(
        MMRE_RECORDS[min_eval_loss_epoch[1]]) + '\n' + str(
        PRED_RECORDS[min_eval_loss_epoch[1]]) + '\n'
    OUTPUT += 'training time: ' + str(time_records[min_eval_loss_epoch[1]]) + '\n'
    OUTPUT += 'Epochs: ' + str(min_eval_loss_epoch[1]) + '\n'
    global BATCH_SIZE
    OUTPUT += 'batch size: ' + str(BATCH_SIZE)

    print("Training completed.")
    print(f"Best epoch: {min_eval_loss_epoch[1]} with eval loss: {min_eval_loss_epoch[0]:.2f}")


WITHIN_PROJECT = True


def main(result_path, files):
    global MODEL, TOKENIZER, MODEL_NAME

    for file in files:
        MODEL = BertForSequence()
        MODEL = MODEL.cuda()
        file_pair, train_dataloader, val_dataloader, all_test_dataloader, test_file_names = data_processing(
            file_pair=file)
        train_eval_test(file_pair, train_dataloader, val_dataloader, all_test_dataloader, MODEL, test_file_names)
        del MODEL
        torch.cuda.empty_cache()
        global OUTPUT
        file_result_path = result_path + file[:-4] + '.txt'
        os.makedirs(os.path.dirname(file_result_path), exist_ok=True)

        if not os.path.exists(file_result_path):
            with open(file_result_path, 'w+') as file:
                file.writelines(OUTPUT)
                OUTPUT = ""
        else:
            with open(file_result_path, 'w') as file:
                file.writelines(OUTPUT)
                OUTPUT = ""

        global projectnum
        projectnum = projectnum + 1


if __name__ == "__main__":
    # define files to be used
    DATA_PATH = r'data/'
    DATA_FILES = ['MESOS', 'USERGRID', 'DATA_MANAGEMENT']

    FILE_TYPE = '.csv'  # FOR WHEN USING MY-SE DATASET
    train_test_file_paths = []

    for folder in DATA_FILES:
        folder_files = [f for f in os.listdir(os.path.join(DATA_PATH, folder)) if f.endswith(FILE_TYPE)]
        folder_files.sort(key=str.lower)  # Optional: sort the file names
        # Append with full path or keep track of folder if needed:

        for f in folder_files:
            train_test_file_paths.append(f"{folder}/{f}")

    # For results using My-SE dataset
    main(result_path='results/', files=train_test_file_paths)

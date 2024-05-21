import pandas, glob
import numpy as np
import evaluate
import torch

from transformers import AutoModelForSequenceClassification, get_scheduler
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

# path to folder that contains flows .csv
csv_path = "output/"

dataset = pandas.concat([pandas.read_csv(f) for f in glob.glob(csv_path + "/*.csv")]).reset_index(drop=True)
print("All .csv loaded into dataframe")
# Dataset: 76 columns
# Columns 0,1 -> label , category
# Last 6 columns: IP_DST, IP_SRC, TOT_BYTES
#                 TOT_PACKETS, TOT_OUT_PACKETS
#                 TOT_IN_PACKETS
# Features Columns: 76-2-6 = 68 columns
X_all_features = np.array(dataset.iloc[:, 2:70])
# 0-68 All Features
X_time_bursts_sizes = np.array(X_all_features[:, ])
# 0-59 Up to Burst Features
X_time_bursts = np.array(X_all_features[:, :-9])
# 0-23 Only time based features
X_time = np.array(X_all_features[:, :-45])
# App Label Column
y = np.array(dataset.iloc[:, 0])
# Classes, labels, counts
classes_tot, classes_label, count_tot = np.unique(y, return_counts=True, return_inverse=True)

dataset_all = TensorDataset(torch.tensor(X_all_features), torch.tensor(classes_label))
dataset_time_bursts = TensorDataset(torch.tensor(X_time_bursts), torch.tensor(classes_label))
dataset_time = TensorDataset(torch.tensor(X_time), torch.tensor(classes_label))

dataloader_all = DataLoader(dataset_all, shuffle=False, batch_size=8)

model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Meta-Llama-3-8B", num_labels = len(classes_tot))

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 4
num_training_steps = num_epochs * len(dataloader_all)
lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

prog_bar = tqdm(range(num_training_steps))

device = torch.device("cuda") if torch.cuda.is_available() else exit(-1)
model.to(device)

model.train()
for epoch in range(num_epochs):
    for x, y in dataloader_all:
        outputs = model(input_ids=x, labels=y)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        prog_bar.update(1)

metric = evaluate.load("accuracy")
model.eval()

import glob
import pandas
import numpy as np
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


#for data scaling 
def scaling(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X-mean) / std

class ClassificationModel(nn.Module):
    def __init__(self, input_size,hidden_sizes, num_class, dropout_prob=0.1):
        super(ClassificationModel, self).__init__()
        self.batch_norm0 = nn.BatchNorm1d(input_size)
        self.layer1 = nn.Linear(input_size, hidden_sizes[0])
        self.batch_norm1 = nn.BatchNorm1d(hidden_sizes[0])
        self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.batch_norm2 = nn.BatchNorm1d(hidden_sizes[1])
        self.layer3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.batch_norm3 = nn.BatchNorm1d(hidden_sizes[2])
        self.layer4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.batch_norm4 = nn.BatchNorm1d(hidden_sizes[3])
        self.layer5 = nn.Linear(hidden_sizes[3], hidden_sizes[4])
        self.batch_norm5 = nn.BatchNorm1d(hidden_sizes[4])
        self.output_layer = nn.Linear(hidden_sizes[4], num_class)
    
        self.dropout = nn.Dropout(p=dropout_prob)
        
        init.xavier_uniform_(self.layer1.weight)
        init.xavier_uniform_(self.layer2.weight)
        init.xavier_uniform_(self.layer3.weight)
        init.xavier_uniform_(self.layer4.weight)
        init.xavier_uniform_(self.layer5.weight)
        init.xavier_uniform_(self.output_layer.weight)
        
    def forward(self, x):
        x = self.batch_norm0(x)
        x = self.layer1(x)
        x = self.batch_norm1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.batch_norm2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = self.layer3(x)
        x = self.batch_norm3(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = self.layer4(x)
        x = self.batch_norm4(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = self.layer5(x)
        x = self.batch_norm5(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = self.output_layer(x)
        
        return x


if __name__ == '__main__':

    # path to folder that contains flows .csv
    csv_path = "./output/"

    # Load data
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
    classes_tot, count_tot = np.unique(y, return_counts=True)

    label_mapping = {
    'dailymotion': 0,
    'facebook': 1,
    'instagram': 2,
    'replaio_radio': 3,
    'skype': 4,
    'spotify': 5,
    'torbrowser_alpha': 6,
    'twitch': 7,
    'utorrent': 8,
    'youtube': 9
    }

    # y 배열의 문자열 레이블을 정수로 매핑
    y = np.array([label_mapping[label] for label in y])

    # 9:1 로 train, test split 
    train_X, test_X, train_y, test_y = train_test_split(X_time_bursts_sizes, y, test_size=0.1, random_state=42)

    #data scaling 
    scaled_train_X = scaling(train_X)
    scaled_test_X = scaling(test_X)

    #numpy array를 tensor로 변환 
    tensor_X_train = torch.from_numpy(scaled_train_X).float()
    tensor_y_train = torch.from_numpy(train_y)
    tensor_X_test = torch.from_numpy(scaled_test_X).float()
    tensor_y_test = torch.from_numpy(test_y)

    #preparing dataloader
    train_dataset = TensorDataset(tensor_X_train, tensor_y_train)
    test_dataset = TensorDataset(tensor_X_test, tensor_y_test)
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Your device is")
    print(device)

    #학습 준비 
    num_class = 10 
    hidden_sizes = [32, 64, 128, 64, 32]
    model_all = ClassificationModel(len(X_time_bursts_sizes[0]), hidden_sizes, num_class).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_all.parameters(), lr=0.0001)

    # 학습
    num_epochs = int(100)
    for epoch in range(num_epochs):
        total_loss_train = 0 
        total_loss_test = 0 
        num_batch_train = 0 
        num_batch_test = 0 
        model_all.train()
        # Forward pass
        for batch_X, batch_y in train_loader:
            num_batch_train += 1
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_y = batch_y.long()
            outputs_train = model_all(batch_X) 
            loss_train = criterion(outputs_train, batch_y)
            total_loss_train += loss_train
        
            # Backward pass 및 최적화
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
        
        model_all.eval()
        with torch.no_grad():
            for batch_X_test, batch_y_test in test_loader:
                num_batch_test += 1
                batch_X_test, batch_y_test = batch_X_test.to(device), batch_y_test.to(device)
                batch_y_test = batch_y_test.long()
                outputs_test = model_all(batch_X_test)
                loss_test =  criterion(outputs_test, batch_y_test)
                total_loss_test += loss_test 
        
        avg_loss_train = total_loss_train / num_batch_train
        avg_loss_test = total_loss_test / num_batch_test
        # 100 에폭마다 로그 출력
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss_train.item():.4f}, Test Loss: {avg_loss_test.item():.4f}')

    
    model_all.eval()
    with torch.no_grad():
        test_predict = model_all(tensor_X_test.to(device))
        
    test_predicted_classes = torch.argmax(test_predict, dim=1)
    test_predicted_classes = test_predicted_classes.cpu().numpy()
    accuracy = np.mean(test_predicted_classes == test_y)
    print("Accuracy: ")
    print(accuracy)
    confusion_matrix(test_predicted_classes, test_y)
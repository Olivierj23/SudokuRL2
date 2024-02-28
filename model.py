import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = "./model_folder"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class SudoCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 81, 3, stride=3, dtype=float)
        self.linear1 = nn.Linear(729, 4096, dtype=float)
        self.linear2 = nn.Linear(4096, 324, dtype=float)
        self.norm1 = nn.LayerNorm((729,), dtype=float)
        ka = 3 // 2
        kb = ka - 1 if 3 % 2 == 0 else ka
        padding_layer = torch.nn.ReflectionPad2d
        self.conv_layers = nn.Sequential(padding_layer((ka,kb,ka,kb)),
                                         nn.Conv2d(1, 512, 3),
                                         nn.BatchNorm2d(512),
                                         nn.ReLU(True),
                                         padding_layer((ka, kb, ka, kb)),
                                         nn.Conv2d(512, 512, 3),
                                         nn.BatchNorm2d(512),
                                         nn.ReLU(True),
                                         padding_layer((ka, kb, ka, kb)),
                                         nn.Conv2d(512, 512, 3),
                                         nn.BatchNorm2d(512),
                                         nn.ReLU(True),
                                         padding_layer((ka, kb, ka, kb)),
                                         nn.Conv2d(512, 512, 3),
                                         nn.BatchNorm2d(512),
                                         nn.ReLU(True),
                                         padding_layer((ka, kb, ka, kb)),
                                         nn.Conv2d(512, 512, 3),
                                         nn.BatchNorm2d(512),
                                         nn.ReLU(True),
                                         padding_layer((ka, kb, ka, kb)),
                                         nn.Conv2d(512, 512, 3),
                                         nn.BatchNorm2d(512),
                                         nn.ReLU(True),
                                         padding_layer((ka, kb, ka, kb)),
                                         nn.Conv2d(512, 512, 3),
                                         nn.BatchNorm2d(512),
                                         nn.ReLU(True),
                                         padding_layer((ka, kb, ka, kb)),
                                         nn.Conv2d(512, 512, 3),
                                         nn.BatchNorm2d(512),
                                         nn.ReLU(True),
                                         padding_layer((ka, kb, ka, kb)),
                                         nn.Conv2d(512, 512, 3),
                                         nn.BatchNorm2d(512),
                                         nn.ReLU(True),
                                         padding_layer((ka, kb, ka, kb)),
                                         nn.Conv2d(512, 512, 3),
                                         nn.BatchNorm2d(512),
                                         nn.ReLU(True),
                                         padding_layer((ka, kb, ka, kb)),
                                         nn.Conv2d(512, 512, 3),
                                         nn.BatchNorm2d(512),
                                         nn.ReLU(True),
                                         padding_layer((ka, kb, ka, kb)),
                                         nn.Conv2d(512, 512, 3),
                                         nn.BatchNorm2d(512),
                                         nn.ReLU(True),
                                         padding_layer((ka, kb, ka, kb)),
                                         nn.Conv2d(512, 512, 3),
                                         nn.BatchNorm2d(512),
                                         nn.ReLU(True),
                                         padding_layer((ka, kb, ka, kb)),
                                         nn.Conv2d(512, 512, 3),
                                         nn.BatchNorm2d(512),
                                         nn.ReLU(True),
                                         padding_layer((ka, kb, ka, kb)),
                                         nn.Conv2d(512, 512, 3),
                                         nn.BatchNorm2d(512),
                                         nn.ReLU(True))


        self.last_conv = nn.Conv2d(512, 9, 1)

    def forward(self, x):
        x = torch.sub(torch.divide(x, 9), 0.5)
        x = self.conv_layers(x)
        x = self.last_conv(x)


        return x

    def save(self, file_name='model.pth'):
        model_folder_path = "./model_folder"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1: predicted Q values with current state
        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action).item()] = Q_new
        # 2: reward update function: r + y *max(next_predicted Q value)

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

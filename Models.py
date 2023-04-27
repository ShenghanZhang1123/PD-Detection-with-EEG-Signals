import torch.nn as nn
import torch
import numpy as np

def shape_cal(dim, pool_dim, stride_dim):
    return np.floor((dim - pool_dim) / stride_dim) + 1

# Define the model
class EEGNet(nn.Module):
    def __init__(self, n_classes=2, n_elec=64, records=1600):
        super(EEGNet, self).__init__()
        self.layer1 = nn.Sequential(  # input size : (sample, 1, n_elec, records)
            # nn.ZeroPad2d((15,15,0,0)),
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(1, 100), stride=(1, 1), padding='same'),
            nn.LeakyReLU()
            )  # output size : (sample, 20, n_elec, records)
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=40, kernel_size=(2, 1), stride=(2, 1), padding=0),
            # output size : (sample, 40, n_elec/2, records)
            nn.BatchNorm2d(40, track_running_stats=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # output size : (sample, 40, n_elec/2**2, records/2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=40, out_channels=80, kernel_size=(1, 40), stride=(1, 1), padding='same'),
            # output size : (sample, 80, n_elec/2**2, records/2)
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
            # output size : (sample, 80, n_elec/2**2, shape_cal(records/2, 4, 4))
        )
        self.layer4 = nn.Sequential(
            # nn.ZeroPad2d((15,15,0,0)),
            nn.Conv2d(in_channels=80, out_channels=160, kernel_size=(2, 4), stride=(2, 4)),
            # output size : (sample, 80, n_elec/2**3, shape_cal(shape_cal(records/2, 4, 4), 4, 4))
            nn.BatchNorm2d(160, track_running_stats=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
            # output size : (sample, 80, n_elec/2**3, shape(shape_cal(records/2, 4, 4), 4, 4)/2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=160, kernel_size=(int(n_elec / 2 ** 3), 1),
                      stride=(int(n_elec / 2 ** 3), 1)),
            # output size : (sample, 160, 1, shape_cal(shape_cal(records/2, 4, 4), 4, 4)/2)
            nn.BatchNorm2d(160, track_running_stats=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
            # output size : (sample, 160, 1, shape_cal(shape_cal(records/2, 4, 4), 4, 4)/6)
        )
        self.classify = nn.Sequential(
            nn.Linear(160 * int(shape_cal(shape_cal(records / 2, 4, 4), 4, 4) / 6), n_classes),
            nn.Dropout(0.2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classify(x)
        return x

# Define the model
class EEGNet_1C(nn.Module):
    def __init__(self, n_classes=2, n_elec=1, records=1600):
        super(EEGNet_1C, self).__init__()
        self.layer1 = nn.Sequential(  # input size : (sample, 1, 1, records)
            # nn.ZeroPad2d((15,15,0,0)),
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(1, 200), stride=(1, 1), padding='same'),
            nn.LeakyReLU(),
        )  # output size : (sample, 20, 1, records)
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=40, kernel_size=(1, 40), stride=(1, 2), padding=0),
            # output size : (sample, 40, 1, shape_cal(records, 40, 2))
            nn.BatchNorm2d(40, track_running_stats=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))  # output size : (sample, 40, 1, shape_cal(records, 40, 2)/4)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=40, out_channels=80, kernel_size=(1, 100), stride=(1, 1), padding='same'),
            # output size : (sample, 80, 1, shape_cal(records, 40, 2)/4)
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
            # output size : (sample, 80, 1, shape_cal(records, 40, 2)/16)
        )
        self.layer4 = nn.Sequential(
            # nn.ZeroPad2d((15,15,0,0)),
            nn.Conv2d(in_channels=80, out_channels=160, kernel_size=(1, 20), stride=(1, 2)),
            # output size : (sample, 160, 1, shape_cal(shape_cal(records, 40, 2)/16, 20, 2))
            nn.BatchNorm2d(160, track_running_stats=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
            # output size : (sample, 160, 1, shape_cal(shape_cal(records, 40, 2)/16, 20, 2)/2)
        )
        self.classify = nn.Sequential(
            nn.Linear(160 * int(shape_cal(shape_cal(records, 40, 2)/16, 20, 2)/2), n_classes),
            nn.Dropout(0.2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classify(x)
        return x

# Define the model
class EEGNet_1C_2(nn.Module):
    def __init__(self, n_classes=2, n_elec=1, records=1600):
        super(EEGNet_1C_2, self).__init__()
        self.layer1 = nn.Sequential(  # input size : (sample, 1, 1, records)
            # nn.ZeroPad2d((15,15,0,0)),
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(1, 200), stride=(1, 1), padding='same'),
            nn.LeakyReLU(),
        )  # output size : (sample, 20, 1, records)
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=40, kernel_size=(1, 40), stride=(1, 2), padding=0),
            # output size : (sample, 40, 1, shape_cal(records, 40, 2))
            nn.BatchNorm2d(40, track_running_stats=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))  # output size : (sample, 40, 1, shape_cal(records, 20, 2)/4)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=40, out_channels=80, kernel_size=(1, 100), stride=(1, 1), padding='same'),
            # output size : (sample, 80, 1, shape_cal(records, 40, 2)/4)
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
            # output size : (sample, 80, 1, shape_cal(records, 40, 2)/16)
        )
        self.layer4 = nn.Sequential(
            # nn.ZeroPad2d((15,15,0,0)),
            nn.Conv2d(in_channels=80, out_channels=160, kernel_size=(1, 20), stride=(1, 2)),
            # output size : (sample, 160, 1, shape_cal(shape_cal(records, 40, 2)/16, 20, 2))
            nn.BatchNorm2d(160, track_running_stats=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
            # output size : (sample, 160, 1, shape_cal(shape_cal(records, 40, 2)/16, 20, 2)/2)
        )
        self.layer5 = nn.Sequential(
            # nn.ZeroPad2d((15,15,0,0)),
            nn.Conv2d(in_channels=160, out_channels=160, kernel_size=(1, 10), stride=(1, 2)),
            # output size : (sample, 160, 1, shape_cal(shape_cal(shape_cal(records, 40, 2)/16, 20, 2)/2, 10, 2))
            nn.BatchNorm2d(160, track_running_stats=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
            # output size : (sample, 160, 1, shape_cal(shape_cal(shape_cal(records, 40, 2)/16, 20, 2)/2, 10, 2)/2)
        )
        self.classify = nn.Sequential(
            nn.Linear(160 * int(shape_cal(shape_cal(shape_cal(records, 40, 2)/16, 20, 2)/2, 10, 2)/2), n_classes),
            nn.Dropout(0.2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classify(x)
        return x
import torch
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.cnn_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 6), stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 6), stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 6), stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.linear_1 = nn.Linear(11264, 128)
        self.drop_1 = nn.Dropout(0.2)
        self.lstm = nn.LSTM(128, 64, num_layers=2, batch_first=True,
                            dropout=0.2, bidirectional=True)
        
        self.output1 = nn.Linear(128, num_classes+1)
        self.output2 = nn.Linear(128, num_classes+1)
        self.output3 = nn.Linear(128, num_classes+1)
        self.output4 = nn.Linear(128, num_classes+1)

    def forward(self, x):
        bs, _, _, _ = x.size()
        x = self.cnn_extractor(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(bs, x.size(1), -1)
        x = F.relu(self.linear_1(x))
        x = self.drop_1(x)
        x, _ = self.lstm(x)


if __name__ == "__main__":
    input_img = torch.rand(10, 3, 700, 300)
    model = Model(10)
    x = model(input_img)
    x = x.permute(0,3,1,2)
    print(x.shape)
    x = x.view(10, x.size(1),-1)
    print(x.shape)

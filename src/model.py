import torch
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.cnn_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 6), stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 6), stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 6), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.linear_1 = nn.Linear(38400, 512)
        self.drop_1 = nn.Dropout(0.2)
        self.lstm = nn.LSTM(
            512, 128, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True
        )

        self.output1 = nn.Linear(256, num_classes + 1)
        self.output2 = nn.Linear(256, num_classes + 1)
        self.output3 = nn.Linear(256, num_classes + 1)
        self.output4 = nn.Linear(256, num_classes + 1)

    def get_loss(self,bs, x, targets):
        log_softmax_values = F.log_softmax(x, 2)
        input_lengths = torch.full(
            size=(bs,),
            fill_value=log_softmax_values.size(0),
            dtype=torch.int32
        )
        # print(input_lengths)
        target_lengths = targets[:,0]
        # print(target_lengths)
        # print(targets[:,0]+1)
        loss = nn.CTCLoss(blank=0)(
            log_softmax_values, targets[:,1:], input_lengths, target_lengths
        )
        return loss

    def forward(self, x, targets=None):
        bs, _, _, _ = x.shape
        # print(x.shape)
        x = self.cnn_extractor(x)
        # print(x.shape)
        x = x.permute(0, 3, 1, 2)
        x = x.view(bs, x.size(1), -1)
        # print(x.shape)
        x = F.relu(self.linear_1(x))
        # print(x.shape)
        x = self.drop_1(x)
        x, _ = self.lstm(x)
        # print(x.shape)
        x1 = self.output1(x)
        x2 = self.output2(x)
        x3 = self.output3(x)
        x4 = self.output4(x)
        if targets is not None: 
            loss1 = self.get_loss(bs, x1.permute(1,0,2), targets['company'])
            loss2 = self.get_loss(bs, x2.permute(1,0,2), targets['address'])
            loss3 = self.get_loss(bs, x3.permute(1,0,2), targets['date'])
            loss4 = self.get_loss(bs, x4.permute(1,0,2), targets['total'])

            return {
                'preds': [x1,x2,x3,x4],
                'losses': [loss1,loss2,loss3, loss4]
            }

        return {
            'preds': [x1,x2,x3,x4]
        }

if __name__ == "__main__":
    import datautils
    import dataset
    import joblib 
    from sklearn.model_selection import train_test_split

    path=datautils.Path('../input/train_data/')
    image_files = datautils.get_images(path)
    train_paths, valid_paths = train_test_split(image_files, test_size=0.3, random_state=42)
    encoder = joblib.load('label_encoder.pkl')
    ds = dataset.Dataset(train_paths, datautils.get_label, encoder,size=(1200,600))
    dl = torch.utils.data.DataLoader(ds, batch_size=10)
    batch = next(iter(dl))
    images = batch.pop('images')
    model = Model(len(encoder.classes_))
    x = model(images, batch)
    print(x['losses'])
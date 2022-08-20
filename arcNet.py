import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
from sklearn.neighbors import KNeighborsClassifier
import pickle

scaler = TfidfVectorizer()

TFIDF_SHAPE = 3657
TFIDF_PATH = 'tfidf.bin'
KNN_PATH = 'knn.bin'
NET_PATH = 'arc_net.pth'


class MyCustomDataset(Dataset):
    def __init__(self,
                 X,
                 y,
                 encoder,
                 scaler = scaler,
                ):
        ## list of tuples: (img, label)
        self._items = []
        self.scaler = scaler
        self.encoder = encoder
        for (text, label) in zip(self.scaler.transform(X).toarray(), y):
            self._items.append((text, self.encoder[label]))        

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        text, label = self._items[index]

        return text, label

class ArcLoss(nn.Module):
    def __init__(self, in_features, out_features):
        super(ArcLoss, self).__init__()
        self.s = 30.0
        self.m = 0.4
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = 1e-7

    def forward(self, x, labels):
        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)
        cos_theta = self.fc(x)
        
        numerator = cos_theta.transpose(0, 1)
        numerator = torch.diagonal(numerator[labels])
        numerator = torch.clamp(numerator, -1+self.eps, 1-self.eps)
        numerator = self.s*torch.cos(torch.acos(numerator)+self.m)
        
        excluded_real_class = torch.cat([
            torch.cat([cos_theta[i, :y], cos_theta[i, y+1:]])[None, :]
            for i, y in enumerate(labels)
        ], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s*excluded_real_class), dim=1)
        
        loss = numerator - torch.log(denominator)
        return cos_theta, -loss.mean()

class ArcNet(pl.LightningModule):
    def __init__(self, emb_net, train_classes):
        super().__init__()
        self.emb_net = emb_net
        self.classifier = ArcLoss(512, train_classes)

    def forward(self, x, cudaaa = True):
        x = x.type(torch.FloatTensor)
        if cudaaa and torch.cuda.is_available():
            x = x.type(torch.cuda.FloatTensor)
        
        output = self.emb_net(x)
        output = self.classifier(output)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        x = x.type(torch.FloatTensor)
        y = y.type(torch.LongTensor)
        if torch.cuda.is_available():
            x = x.type(torch.cuda.FloatTensor)
            y = y.type(torch.cuda.LongTensor)
            
        outputs = self.emb_net(x)
        outputs, loss = self.classifier(outputs, y)
        acc = (outputs.argmax(dim=1) == y).sum().item() / len(outputs)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

class EmbeddingNet(nn.Module):
    def __init__(self, input_shape=TFIDF_SHAPE):
        super(EmbeddingNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_shape, 2048),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Dropout(p=0.6),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(negative_slope=0.15),
            nn.Dropout(p=0.4),
            nn.Linear(1024, 512)
        )

    def forward(self, x, cudaaa = True):
        x = x.type(torch.FloatTensor)
        if cudaaa and torch.cuda.is_available():
            x = x.type(torch.cuda.FloatTensor)
            
        output = self.net(x)
        return output

def train_knn(emb_net, dl_train, is_normalize=False):
    data_x = []
    data_y = []
    with torch.no_grad():
        for data in dl_train:
            inputs, labels = data[0], data[1]
            feats = emb_net(inputs, cudaaa=False).detach().cpu().numpy()
            
            data_x.append(feats)
            data_y.append(labels)
    data_x = np.concatenate(data_x, axis=0)
    data_y = np.concatenate(data_y, axis=0)
    
    knn = KNeighborsClassifier(n_neighbors=1, metric='cosine') 
    knn = knn.fit(data_x, data_y)
    
    return knn

def test_knn(emb_net, knn, dl_test, is_normalize=False):
    total_correct = 0
    total_cnt = 0
    with torch.no_grad():
        for data in dl_test:
            inputs, labels = data[0], data[1].detach().cpu().numpy()
            feats = emb_net(inputs, cudaaa=False).detach().cpu().numpy()
            preds = knn.predict(feats)
            
            total_correct += (preds == labels).sum()
            total_cnt += len(preds)
    
    acc = total_correct/total_cnt
    print(f'Accuracy = {acc*100}%')
    return acc

def prepare_data(
        train_path: str = 'train.csv', 
        val_path: str = 'val.csv', 
        test_path: str = 'test.csv'
        ):
    train_t = pd.read_csv(train_path)
    val_t = pd.read_csv(val_path)
    test_t = pd.read_csv(test_path)

    Train = train_t
    Val = val_t
    Test = test_t

    X_train, y_train = Train['merged'].to_numpy(), Train['reference_id'].to_numpy()
    X_val, y_val = Val['merged'].to_numpy(), Val['reference_id'].to_numpy()
    X_test, y_test = Test['merged'].to_numpy(), Test['reference_id'].to_numpy()

    encoder = {i: k for k, i in enumerate(np.unique(y_train))}

    scaler.fit(X_train)

    return X_train, y_train, X_val, y_val, X_test, y_test, encoder

def train_model(X_train, X_val, y_train, y_val, encoder):
    L_train = np.stack([*X_train, *X_val], axis=0)
    Ly_train = np.stack([*y_train, *y_val], axis=0)

    dL_train = MyCustomDataset(L_train, Ly_train, encoder=encoder)
    arc_train_dl1 = DataLoader(dL_train, 32, num_workers=0)

    net_arc = ArcNet(EmbeddingNet(), train_classes=len(encoder.keys()))
    trainer = pl.Trainer(gpus=0, max_epochs=5)
    trainer.fit(net_arc, arc_train_dl1)

    return net_arc

def dataLoaderData(X_train, X_val, X_test, y_train, y_val, y_test, encoder):
    ds_train = MyCustomDataset(X_train, y_train, encoder=encoder)
    ds_val = MyCustomDataset(X_val, y_val, encoder=encoder)
    ds_test = MyCustomDataset(X_test, y_test, encoder=encoder)

    dl_train = DataLoader(ds_train, batch_size=32, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=32, shuffle=False, num_workers=0)
    dl_test = DataLoader(ds_test, batch_size=32, shuffle=False, num_workers=0)

    return dl_train, dl_val, dl_test


def train_head(net_arc, dl_train, dl_test, test: bool = True):
    net_arc.eval()
    net_arc.emb_net.to('cpu')
    knn = train_knn(net_arc.emb_net, dl_train, is_normalize=True)
    if test:
        test_knn(net_arc.emb_net, knn, dl_test, is_normalize=True)

    return knn

def save_net(net, path: str):
    torch.save(net.state_dict(), path)

def load_net(encoder, path: str = NET_PATH):
    test_net = ArcNet(EmbeddingNet(), train_classes=len(encoder.keys()))
    test_net.load_state_dict(torch.load(path))

    return test_net

def save_sklearn_model(model, path: str):
    modelFile = open(path, 'wb') 
    pickle.dump(model, modelFile) 

def load_sklearn_model(path: str):
    model = pickle.load(open(path, 'rb'))

    return model

if __name__=='__main__':
    if sys.argv[1] == 'train':
        X_train, y_train, X_val, y_val, X_test, y_test, encoder = prepare_data()

        net = train_model(X_train, X_val, y_train, y_val, encoder)

        dl_train, dl_val, dl_test = dataLoaderData(X_train, X_val, X_test, y_train, y_val, y_test, encoder)
        
        head = train_head(net, dl_train, dl_test)

        save_sklearn_model(scaler, TFIDF_PATH)
        save_sklearn_model(head, KNN_PATH)
    elif sys.argv[1] == 'test':
        X_train, y_train, X_val, y_val, X_test, y_test, encoder = prepare_data()
        net = load_net(encoder)
        net.eval()
        scaler = load_sklearn_model(TFIDF_PATH)
        head = load_sklearn_model(KNN_PATH)

        dl_train, dl_val, dl_test = dataLoaderData(X_train, X_val, X_test, y_train, y_val, y_test, encoder)
        
        test_knn(net.emb_net, head, dl_test, is_normalize=True)

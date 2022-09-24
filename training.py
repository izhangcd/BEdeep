# -*- encoding: utf-8 -*-


import argparse
import pkbar
import torch

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import torch.optim as optim

from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import Dataset, DataLoader

from utils import EarlyStopping


# reciprocal of offtype ratio in offtarget library
ABE_OFFTYPE = {'1del': 15.976929, '1ins': 10.902291, '1mis': 1.810554, 
    '2del': 45.733515, '2ins': 100.725625, '2mis': 7.646374, '3mis': 38.73471, 
    '4mis': 37.740532, '5mis': 36.532119, '6mis': 42.461986, 'N': 340.419844, 
    'Y': 69.54169, 'mix': 97.659573}
CBE_OFFTYPE = {'1del': 16.833737, '1ins': 10.03318, '1mis': 1.961475, 
    '2del': 47.223878, '2ins': 99.985738, '2mis': 6.869565, '3mis': 28.927365, 
    '4mis': 34.587272, '5mis': 29.653575, '6mis': 34.954178, 'N': 216.021547, 
    'Y': 77.686015, 'mix': 90.546468}

# ABE_OFFTYPE = {'2ins':10,'2del':10,'2mis':5,'mix':10,'1del':2,'1ins':2,
#        '1mis':1,'3mis':10,'4mis':1,'5mis':1,'6mis':5,'N':1,'Y':1,'7mis':1,'8mis':1,'9mis':1}
# CBE_OFFTYPE = {'2ins':10,'2del':10,'2mis':5,'mix':10,'1del':2,'1ins':2,
#        '1mis':1,'3mis':10,'4mis':1,'5mis':1,'6mis':5,'N':1,'Y':1,'7mis':1,'8mis':1,'9mis':1}

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def do_encoding(source, target, offtype, is_ABE=True):
    '''Sequence encoding.'''
    alignments = pairwise2.align.globalms(source, target, 1, -1, -3, -2)
    source, _aln, target = format_alignment(*alignments[0]).split('\n')[:-2]
    encode_dict = {'<pad>':0, 'A': 1, 'C': 2, 'G':3, 'T': 4, '-': 5}
    encode_list = []
    for i, j in zip(source, target):
        encode_list.append(encode_dict[i])
        encode_list.append(encode_dict[j])
    offset_list = list(range(0, len(encode_list), 2))
    if is_ABE:
        return source, target, encode_list, offset_list, ABE_OFFTYPE[offtype]
    else:
        return source, target, encode_list, offset_list, CBE_OFFTYPE[offtype]


class gRNADataset(Dataset):
    def __init__(self, df, is_ABE=True):
        df[['source', 'target', 'encode', 'offset', 'offtype']] = df.apply(
            lambda x: do_encoding(x['source'], x['target'], x['offtype'], 
            is_ABE), axis=1, result_type='expand')
        df.reset_index(drop=True, inplace=True)
        self.source = df['source']
        self.target = df['target']
        self.offtype = df['offtype'].values
        self.efficiency = df['efficiency'].values
        self.encode = df['encode'].values
        self.offset = df['offset'].values
        print(f'Finished loading the {data} ({len(df)} samples found)')

    def __len__(self):
        return len(self.source)
    
    def __getitem__(self, index):
        source = self.source[index] 
        target = self.target[index]
        offtype = self.offtype[index]
        efficiency = torch.FloatTensor(np.array(self.efficiency[index]))
        encode = torch.LongTensor(self.encode[index])
        offset = torch.LongTensor(self.offset[index])
        return source, target, offtype, efficiency, encode, offset


def generate_batch(batch):
    source_list = []
    target_list = []
    offtype_list = []
    efficiency_list = []
    encode_list = []
    offset_list = []
    seqlen_list = []
    old_max_ofs = 0
    batch = [(a, b, c, d, e, f, f.shape[0])  for a, b, c, d, e, f in 
        sorted(batch, key=lambda x: x[5].shape[0], reverse=True)]
    for i, (a, b, c, d, e, f, g) in enumerate(batch):
        source_list.append(a)
        target_list.append(b)
        offtype_list.append(c)
        efficiency_list.append(d)
        encode_list += e
        i = 1 if i > 0 else 0
        offset = old_max_ofs + f + i * 2
        old_max_ofs = offset[-1]
        offset_list += list(offset)
        seqlen_list.append(g)
    return (source_list, target_list, torch.FloatTensor(offtype_list), 
            torch.FloatTensor(efficiency_list), torch.LongTensor(encode_list), 
            torch.LongTensor(offset_list), torch.LongTensor(seqlen_list))


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.EmbeddingBag(input_dim, emb_dim)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, n_layers, bidirectional=True)
        # self.norm2 = nn.LayerNorm(6 * enc_hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_feat = nn.Linear(6 * enc_hid_dim, 3 * enc_hid_dim)
        self.norm3 = nn.LayerNorm(3 * enc_hid_dim)
        self.fc_out = nn.Linear(3 * enc_hid_dim, 1)

    def forward(self, seq, offset, length):
        global debug_var
        emb = self.dropout(self.embedding(seq, offset))
        emb = self.norm1(emb)

        #根据开始位置确定结束位置
        l_b = length.cumsum(dim=0)

        max_len = length[0]
        start_tmp = 0
        emb_tensors = []
        for i in range(len(length)):
            start = start_tmp
            end = l_b[i]
            emb_new = emb[start:end]
            seq_len = emb_new.shape[0]
            if max_len-seq_len>0:
                zeros = torch.zeros(max_len - seq_len, emb.shape[1]).to(device)
                emb_new = torch.cat([emb_new, zeros])
            emb_tensors.append(emb_new)
            start_tmp = end
            
        emb_tensor = torch.stack(emb_tensors).transpose(1, 0)
        embed_input_x_packed = rnn_utils.pack_padded_sequence(emb_tensor, 
            length.cpu(), batch_first=False)

        pack_out, (hidden,_) = self.rnn(embed_input_x_packed)
    
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
    
        pad_out, _ = rnn_utils.pad_packed_sequence(pack_out)
        
        avg_pool = torch.mean(pad_out.permute(1, 0, 2), 1)
        max_pool, _ = torch.max(pad_out.permute(1, 0, 2), 1)
        x = torch.cat([hidden, avg_pool, max_pool], dim=1)
        # x = self.norm2(x)
        x = self.dropout(F.relu(self.fc_feat(x)))
        x = self.norm3(x)
        fc_out = self.fc_out(x)
        return fc_out


def get_pred(iter_, model):
    model.eval()
    lst_dfs = []
    with torch.no_grad():
        for i, batch in enumerate(iter_):
            seq1 = batch[4].to(device)
            seq2 = batch[5].to(device)
            y = batch[3].unsqueeze(1).to(device)
            length = batch[6].to(device)

            out_eff = model(seq1, seq2, length)
            #前面换成了 100，这里要抵消
            y = list(y.view(-1).cpu().numpy()/100)
            #sigmoid 只能是0-1
            out_eff = torch.sigmoid(out_eff)
            out_eff = list(out_eff.view(-1).cpu().numpy())

            #获取gRNA 编辑结果的预测活性分布
            df_gRNA = pd.DataFrame({'source': batch[0],'target': batch[1]})
            df_gRNA['y'] = y
            df_gRNA['y_pred'] = out_eff
            lst_dfs.append(df_gRNA)
    df_conc = pd.concat(lst_dfs)
    return df_conc


def init_weights(m):
    for name, param in m.named_parameters():
        if 'rnn.weight_' in name:
            nn.init.orthogonal_(param.data)
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
            

model = Encoder(6, 256, 512, 1, 0.5).to(device)
# model.apply(init_weights)

lr_dic = {0: 0.001, 1: 0.0001, 2: 0.00001, 3: 0.000001, 4: 0.0000001}
my_optim = optim.Adam(model.parameters(),lr=lr_dic[0])
criterion_mse = nn.MSELoss(reduction='none')


def train_model(train_iter, val_iter, patience, j, be):
    global my_optim
    global lr_dic
    N_EPOCHS = 250
    CLIP = 1
    best_valid_loss = float('inf')
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    
    # initialize the early_stopping object
    path = './Models/ABE_checkpoint.pt' if be == 'ABE' else './Models/CBE_checkpoint.pt'
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=path)
    early_num = 0
    for epoch in range(N_EPOCHS):

        train_per_epoch = len(train_iter)
        ################################### Initialization ########################################
        kbar = pkbar.Kbar(target=train_per_epoch, epoch=epoch, num_epochs=N_EPOCHS, width=8, always_stateful=False)
        # By default, all metrics are averaged over time. If you don't want this behavior, you could either:
        # 1. Set always_stateful to True, or
        # 2. Set stateful_metrics=["loss", "rmse", "val_loss", "val_rmse"], Metrics in this list will be displayed as-is.
        # All others will be averaged by the progbar before display.
        ###########################################################################################

        model.train()
        epoch_loss_ = 0
        
        for i, batch in enumerate(train_iter):
            seq1 = batch[4].to(device)
            seq2 = batch[5].to(device)
            y = batch[3].unsqueeze(1).to(device)
            length = batch[6].to(device)
            wgts = batch[2].to(device)

            my_optim.zero_grad()
            outputs = model(seq1, seq2, length)

            #src_y regression
            y_hat = torch.sigmoid(outputs)*100
            loss = (criterion_mse(y, y_hat)*wgts).mean()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            
            my_optim.step()
            
            
            epoch_loss_ += loss.item()
            train_losses.append(loss.item())
            ############################# Update after each batch ##################################
            kbar.update(i, values=[("lr", my_optim.defaults['lr']),("loss", epoch_loss_ / (i + 1))])
            ########################################################################################
        #my_lr_scheduler.step()
        
        model.eval()
        epoch_loss_ = 0
        with torch.no_grad():

            for i, batch in enumerate(val_iter):

                seq1 = batch[4].to(device)
                seq2 = batch[5].to(device)
                y = batch[3].unsqueeze(1).to(device)
                length = batch[6].to(device)
                wgts = batch[2].to(device)

                my_optim.zero_grad()
                outputs = model(seq1, seq2, length)

                #src_y regression
                y_hat = torch.sigmoid( outputs )*100
                loss = (criterion_mse(y, y_hat)).mean()

                #loss = (loss1 + loss2).mean()
                valid_losses.append(loss.item())
                epoch_loss_ += loss.item()
            ################################ Add validation metrics ###################################
            kbar.add(1, values=[("val_loss", epoch_loss_ / len(val_iter))])
            ###########################################################################################        
        valid_loss = epoch_loss_ / len(val_iter)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            p = './Models/ABE.pt' if be == 'ABE' else './Models/CBE.pt'
            torch.save( model.state_dict(), p)     

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model,p)
        
        if early_stopping.early_stop:
            
            early_num = early_num + 1
            if early_num == 5:
                model.load_state_dict(torch.load(p))
                my_optim = optim.Adam(model.parameters(),lr=lr_dic[0])
                early_stopping = EarlyStopping(patience=patience, verbose=True, path=path)
                print('Early Stopping...')
                break
            print("Change learning rate")
            model.load_state_dict(torch.load(p))
            my_optim = optim.Adam(model.parameters(),lr=lr_dic[early_num]) 
            early_stopping = EarlyStopping(patience=patience, verbose=True, path=path)
    # load the last checkpoint with the best model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model training for ABEmax and AncBE4max")
    parser.add_argument("-b", "--base-editor", choices=["ABE", "CBE"], required=True, help="Set base editor model")
    args = parser.parse_args()
    be = args.base_editor
    if be == 'ABE':
        data = './Data/ABEdeepoff.txt'
    else:
        data = './Data/CBEdeepoff.txt'
    df = pd.read_csv(data, sep='\t')
    df['efficiency'] = df['efficiency'] * 100

    train_test_index = ShuffleSplit(n_splits=10, test_size=0.1, 
                                    random_state=2345).split(df)
    train_test_data = []
    for _i, (train_index, test_index) in enumerate(train_test_index):
        train_data, test_data = df.loc[train_index], df.loc[test_index]

        train_dataset = gRNADataset(train_data)
        train_iter = DataLoader(train_dataset, batch_size=32, shuffle=False,
                                collate_fn=generate_batch)

        test_dataset = gRNADataset(test_data)
        test_iter = DataLoader(test_dataset, batch_size=32, shuffle=False,
                            collate_fn=generate_batch)
        train_test_data.append((train_iter, test_iter))


    for i, data in enumerate(train_test_data):
        model.apply(init_weights)
        train_iter, val_iter = data
        train_model(train_iter, val_iter, 5, i, be)

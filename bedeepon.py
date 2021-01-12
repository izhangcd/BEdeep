import argparse
import io
from pathlib import Path
import pickle

import re
import random
import numpy as np
import pandas as pd
import math
from collections import defaultdict, Counter


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,Sampler,BatchSampler,SubsetRandomSampler
import torch.nn.utils.rnn as rnn_utils

#Set computing environment
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

torch.set_printoptions(precision=6,sci_mode=False)
pd.set_option('display.float_format',lambda x : '%.6f' % x)

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

stoi = {'<pad>':0, 'a':1,  'c':2,  'g':3,  't':4, 'A':1, 'C':2, 'G':3, 'T':4, '-':5}
itos = {0: '<pad>', 1:'a',  2:'c',  3:'g',  4:'t', 1:'A', 2:'C', 3:'G', 4:'T', 5:'-'}


import argparse
parser = argparse.ArgumentParser(description="BEdeepon for ABEmax and AncBE4max")
parser.add_argument("-b", "--base-editor", 
                    choices=["ABE", "CBE"],
                    required=True,
                    help="set base editor model")
parser.add_argument("-i","--input-file",help="set input tsv file")
args = parser.parse_args()


print("Preparing computing environment...")
# Model configuration

ENC_INPUT_DIM = 6
ENC_EMB_DIM = 64
ENC_HID_DIM = 128
N_LAYERS = 1
ENC_DROPOUT = 0.5

class Net(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, n_layers, dropout):
        super().__init__()
        
        self.enc_hid_dim = enc_hid_dim
        
        self.embedding = nn.EmbeddingBag(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, n_layers, bidirectional = True)
        #self.attention_layer = Attention(2*enc_hid_dim, seq_len)
        self.dropout = nn.Dropout(dropout)
        #self.fc_feat = nn.Linear(4*enc_hid_dim,2*enc_hid_dim)
        self.fc_out = nn.Linear(2*enc_hid_dim, 1)
        
    def forward(self, seq, offset, length):
        batch_size = len(length)

        emb = self.dropout(self.embedding(seq, offset))
        emb_v = emb.view(batch_size,23, -1)
        emb_vt = emb_v.transpose(1,0)
        
        out, (hidden, _) = self.rnn(emb_vt)
        x = torch.cat( (hidden[-2,:,:], hidden[-1,:,:]), dim = 1 )
        out = F.leaky_relu(self.fc_out(x))  
        return out
    
def get_model():
    abe_model_file = './Models/ABEdeepon.pt'
    cbe_model_file = './Models/CBEdeepon.pt'
    model1 = Net(ENC_INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, N_LAYERS, ENC_DROPOUT).to(device)
    model1.load_state_dict( torch.load(abe_model_file,map_location=torch.device(device)))
    model2 = Net(ENC_INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, N_LAYERS, ENC_DROPOUT).to(device)
    model2.load_state_dict( torch.load(cbe_model_file,map_location=torch.device(device)))
        
    return model1,model2


#Data Preperation
class gRNADataset(Dataset):
    def __init__(self,datafrm):
        self.df = datafrm.reset_index(drop=True)
        self.df.columns = ['seq1','seq2','is_edit']
        self.indexes = list(self.df['seq1'].values)
        self.offsets = list(self.df['seq2'].values)
        self.is_edit = list(self.df['is_edit'].values)
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        indexes = self.indexes[idx]
        offsets = self.offsets[idx]
        is_edit = self.is_edit[idx]
        return indexes, offsets,is_edit
    
    
def generate_batch(batch):
    indexes = []
    offsets = []
    seq_len = []
    edit_status = []
    old_max_ofs = 0
    for i,( idx, ofs, is_edit ) in enumerate(batch):
        indexes += idx
        i = 1 if i > 0 else 0
        offset = old_max_ofs + np.array(ofs) + i * 2
        old_max_ofs = offset[-1]
        offsets += list(offset)
        seq_len.append(len(ofs))
        edit_status.append(is_edit)
        
    a = indexes
    b = offsets
    d = seq_len
    e = edit_status.index(0) #non_id
    return a,b,d,[e]


class SubsetRandomSampler(Sampler):
    # r"""Samples elements randomly from a given list of indices, without replacement.
    # Arguments:
    #     indices (sequence): a sequence of indices
    # """

    def __init__(self, lst_inds):
        self.lst_idx = lst_inds

    def __iter__(self):
        idx = [self.lst_idx[i] for i in torch.randperm(len(self.lst_idx))]
        return iter(idx)

    def __len__(self):
        return len(self.indices)
    
    
class BatchSampler(Sampler):
    '''
    grp must be reset index first
    '''
    # 批次采样
    def __init__(self,gRNAs, grp):
        lst_idx = []
        for gRNA in gRNAs:
            lst_idx.append(grp[gRNA].index.values)
        sampler = SubsetRandomSampler(lst_idx)
        self.sampler = sampler
        self.batch_size = len(grp[gRNA])
    def __iter__(self):
        for idx in self.sampler:
            self.batch_size = len(idx)
            yield idx

    def __len__(self):
        return self.batch_size        
    
    
def get_encoding(row):
    seq1 = row['source']
    seq2 = row['target']
    is_edit = row['is_edit']
    seq_idx = []
    for i,(nuc1, nuc2) in enumerate(zip(seq1,seq2)):
        seq_idx += [stoi[nuc1]] + [stoi[nuc2]]
    offset = [i for i in range(0,len(seq_idx),2)]
    return seq_idx, offset, is_edit


## create precomputed outcome combinations 
from itertools import permutations,combinations,product

def get_comb_df(seq, is_ABE=True):
    seq = seq.upper()
    core_l_pos = [i for i in range(2)]
    core_pos = [i for i in range(2,17)]
    core_r_pos = [i for i in range(17,20)]

    core_ = 0
    ext_l = 0
    ext_r = 0
    
    nt_cols = get_nt_cols( seq, is_ABE)
    for c in nt_cols:
        if int(c[1:]) in core_pos:
            core_ += 1
        elif int(c[1:]) in core_l_pos:
            ext_l += 1
        else:
            ext_r += 1

    #core_left construct
    lst_ext_l = get_ext_comb(ext_l, is_ABE)
    lst_ext_r = get_ext_comb(ext_r, is_ABE)
    lst_core = get_core_comb(core_, is_ABE)

    lst_ = []
    for i in lst_ext_l:
        for j in lst_core:
            for k in lst_ext_r:
                lst_.append(i + j + k) 
    df_comb = pd.DataFrame(lst_, columns=nt_cols)

    col_pos = [int(col[1:]) for col in nt_cols]
    def process(row):
        lst_seq = list(seq)
        for i, pos in enumerate(col_pos):
            lst_seq[pos] = row.values[i]
        return ''.join(lst_seq)
    df_comb['target'] = df_comb.apply(lambda x:process(x),axis=1)    
    return df_comb

def get_ext_comb(ext_, is_ABE=True):
    
    editing = ('A','G') if is_ABE else ('C','T')
    
    items = [editing[1] if i == 0 else editing[0] for i in range(ext_)]
    lst_ext = []
    for c in permutations(items, ext_):
        lst_ext.append(c)
    lst_ext.append(tuple([editing[0] for i in range(ext_)])) 
    lst_ext = list(set(lst_ext))   
    return lst_ext

def get_core_comb(core_, is_ABE=True):
    editing = ('A','G') if is_ABE else ('C','T')
    lst_core = []
    for output in product(editing, repeat=core_):
        lst_core.append(output)    
    return lst_core

def get_nt_cols(seq, is_ABE=True):
    editing = ('A','G') if is_ABE else ('C','T')
    sub_nuc = editing[0]
    nt_cols = []
    for pos in range( len( seq ) ):
        ref_nt = seq[pos]
        nt_col = f'{ref_nt}{pos}'
        if nt_col in [f'{sub_nuc}{i}' for i in range(0,20)]:
            nt_cols.append( nt_col )
    return nt_cols

def get_nuc_info(batch):
    seqs = batch[0]
    length = len(batch[2])
    lst_seqs = np.array_split(seqs,length)
    lst_seq = []
    for seq in lst_seqs:
        seq1 = []
        seq2 = []    
        for i,idx in enumerate(seq):
            if (i + 1) % 2 == 0:
                seq2.append(seq[i])
            else:
                seq1.append(seq[i])
        seq1 = ''.join([itos[i] for i in seq1])
        seq2 = ''.join([itos[i] for i in seq2])
        lst_seq.append((seq1,seq2))
        
    return lst_seq

def generate_pred_batch(batch):
    indexes = []
    offsets = []
    seq_len = []
    edit_status = []
    old_max_ofs = 0
    for i,( idx, ofs, is_edit) in enumerate(batch):
        indexes += idx
        i = 1 if i > 0 else 0
        offset = old_max_ofs + np.array(ofs) + i * 2
        old_max_ofs = offset[-1]
        offsets += list( offset )
        seq_len.append( len(ofs) )
        edit_status.append( is_edit )
        
    a = indexes
    b = offsets
    c = seq_len
    d = edit_status.index(0) #non_id
    return a, b, c, [d]


#### 内部验证数据集
def get_pred_y(batch,model):
    model.eval()
    seq1 = torch.LongTensor(batch[0]).to(device)
    seq2 = torch.LongTensor(batch[1]).to(device)
    length = torch.LongTensor(batch[2]).to(device)
    non_id = torch.LongTensor(batch[3]).to(device)

    with torch.no_grad():

        outputs = model(seq1, seq2, length)
        
        #src_y regression
        outs = outputs.view(-1)
        y_hat_sm = F.softmax(outs,0)
        y_pred = list(1 - y_hat_sm[non_id].view(-1).cpu().numpy())

        df_gRNA = pd.DataFrame(get_nuc_info(batch),columns=['target','outcome'])
        df_gRNA['y_pred'] = list(F.softmax(outputs,0).squeeze(1).cpu().numpy())
    return y_pred[0],df_gRNA


def do_pred(endo_iter,model):
    lst_pred = []
    lst_dfs = []
    lst_eff = []
    for item in endo_iter:
        eff_pred, df = get_pred_y( item, model )
        lst_eff.append([df.target.unique()[0], eff_pred])
        lst_dfs.append(df)
    df_eff = pd.DataFrame( lst_eff, columns=[ 'target','eff_pred'] )    
    df_lst = pd.concat(lst_dfs)
    return df_eff,df_lst

def prep_inputs(df_inputs,is_ABE=True):
    lst_gRNA = []
    for gRNA in df_inputs.source.unique():
        df_gRNA = get_comb_df( gRNA, is_ABE )[['target']]
        df_gRNA['source'] = gRNA.upper()
        df_gRNA['is_edit'] = np.where( df_gRNA.source == df_gRNA.target, 0, 1 )
        df_gRNA[[ 'seq1','seq2','is_edit' ]] = df_gRNA.apply( lambda x:pd.Series( get_encoding(x) ),axis=1 )
        lst_gRNA.append(df_gRNA)
        df_ = pd.concat( lst_gRNA ).reset_index( drop=True )
        grp_ = dict( list( df_.groupby('source') ) )
    dataset = gRNADataset( df_[['seq1','seq2','is_edit']] )
    batches =  DataLoader(dataset, batch_sampler=BatchSampler(grp_.keys(), grp_), collate_fn=generate_batch) 
    return batches

print("Load model...")
model_on = get_model()

def main():
    input_file = Path(args.input_file)
    stem = input_file.stem
    output_dir = utils.safe_makedir(input_file.parent / 'on_output')
    eff_file = input_file.parent / 'on_output' / (str(input_file.stem) + '_eff.csv')
    outcomes_file = input_file.parent / 'on_output' / (str(input_file.stem) + '_outcomes.csv')

    if not input_file.is_file():
        exit("File doesn't exist!")
    else:
        print("Load input file...")
        try:
            df_inputs = pd.read_csv(input_file)
            df_inputs.columns = ['source']
            #前端 base_editor 1: ABE ，0 ：CBE
            model = model_on[0] if args.base_editor == "ABE" else model_on[1]
            is_ABE = True if args.base_editor == "ABE" else False
            print("Do prediction...")
            batches = prep_inputs( df_inputs, is_ABE=base_editor )
            df_eff,df_outcomes = do_pred( batches, model )
            res_eff = df_eff.sort_values( by = 'eff_pred', ascending = False ).reset_index( drop=True )
            res_outcome = df_outcomes.reset_index( drop=True )
            res_eff.to_csv(eff_file)
            res_outcome.to_csv(outcomes_file)
            print("Finished prediciton!")
        except Exception as e:
            print(str(text))
if __name__ == "__main__":
    main() 
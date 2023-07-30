# -*- encoding: utf-8 -*-


import argparse

import random
import numpy as np
import pandas as pd
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils

from Bio import pairwise2
from Bio.pairwise2 import format_alignment


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


SEED = 1356
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ENC_INPUT_DIM = 6
ENC_EMB_DIM = 256
ENC_HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5

class Net(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super(Net, self).__init__()
        
        self.embedding_1 = nn.Embedding(input_dim, emb_dim)
        self.embedding_2 = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(input_size=emb_dim, hidden_size=hid_dim, num_layers=n_layers,
                           bidirectional=True)
        self.dropout = nn.Dropout(dropout)     
        self.fc_feat_1 = nn.Linear(6 * hid_dim, 3 * hid_dim)
        self.fc_out = nn.Linear(3 * hid_dim, 1) 
        self.att_score = None
    
    def attention_net(self, x, query, mask=None): 
        
        d_k = query.size(-1)
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  
        alpha_n = F.softmax(scores, dim=-1) 
        context = torch.matmul(alpha_n, x).sum(1)
        return context, alpha_n
    
    def forward(self,seq_1, seq_2):
        global debug_mod_var
        emb_1 = self.embedding_1(seq_1)
        emb_2 = self.embedding_2(seq_2)
        emb_comb = self.dropout(emb_1 + emb_2)
        debug_mod_var = seq_1, seq_2,emb_1,emb_2
        #self.embedding = emb_comb
        out, (hid_, _) = self.rnn(emb_comb)
        hidden = torch.cat( (hid_[-2,:,:], hid_[-1,:,:]), dim = 1 )
        
        out = out.permute(1,0,2)
        avg_pool = torch.mean( out, 1)
        max_pool, _ = torch.max( out, 1)
                             
        query = self.dropout(out)
        # 加入attention机制
        attn_output, alpha_n = self.attention_net(out, query)
        self.att_score = alpha_n
        
        #hid_size*2*3
        x = torch.cat([ attn_output, hidden, max_pool], dim=1)
        x = self.dropout(F.relu(self.fc_feat_1( x )))
        fc_out = self.fc_out(x)
        return fc_out
    

class gRNADataset(Dataset):
    def __init__(self, df, is_ABE=True):
        df[['seq1', 'seq2']] = df.apply(
            lambda x: do_encoding(x['source'], x['target']), axis=1, result_type='expand')
        df.reset_index(drop=True, inplace=True)
        self.source = df['source']
        self.target = df['target']
        self.efficiency = df['efficiency'].values
        self.seq1 = df['seq1'].values
        self.seq2 = df['seq2'].values
        self.otype = df['type']
        # print(f'Finished loading the {data} ({df.shape[0]} samples found)')

    def __len__(self):
        return len(self.source)
    
    def __getitem__(self, index):
        source = self.source[index] 
        target = self.target[index]
        y = torch.FloatTensor(np.array(self.efficiency[index]))
        seq1 = torch.LongTensor(self.seq1[index])
        seq2 = torch.LongTensor(self.seq2[index])
        seq_len = seq1.shape[0]
        otype = self.otype[index]
        return source, target, y, seq1, seq2, seq_len, otype
    

#seq_idx, offset, y, a, c,off_type
def generate_batch(batch):
    global debug_var
    ys = []
    seqlen_lst = []
    source_lst = []
    target_lst = []
    seq1_lst = []
    seq2_lst = []
    otype_lst = []
    #x[-2]即seq1，他的shape[0]就是seq1的长度（同时也是seq2）的长度
    #通过对seq1长度进行排序，可以令每一个batch中都是第一个序列长度最长；
    batch = [ (a, b, c, d, e, f, g) for a, b, c, d, e, f, g in sorted( batch, key=lambda x:x[-2], reverse=True) ]

    for i, (source, target, y, seq1, seq2, seq_len, otype) in enumerate(batch):
        source_lst.append(source)
        target_lst.append(target)
        ys.append(y)
        
        seq1_lst.append(seq1)
        seq2_lst.append(seq2)
        seqlen_lst.append(seq_len)
        otype_lst.append(otype)
    
    debug_var = seq1_lst, seq2_lst
    # 将序列填充到相同的长度，并设置填充的值为0
    padded_seqs = rnn_utils.pad_sequence(seq1_lst, batch_first=False, padding_value=0)
    # 对于每个序列，通过 mask 将填充的部分设置为-1
    mask = padded_seqs.ne(0)
    seq1_batch = padded_seqs.masked_fill(~mask, 0)
    
    padded_seqs = rnn_utils.pad_sequence(seq2_lst, batch_first=False, padding_value=0)
    # 对于每个序列，通过 mask 将填充的部分设置为-1
    mask = padded_seqs.ne(0)
    seq2_batch = padded_seqs.masked_fill(~mask, 0)
    
    return (source_lst, target_lst, 
            torch.FloatTensor(ys),
            seq1_batch, seq2_batch,
            torch.LongTensor(seqlen_lst), otype_lst)  


def do_encoding(source, target):
    aln = pairwise2.align.globalms(source, target, 1, -1, -3, -2)
    src, _aln, tgt = format_alignment(*aln[0]).split('\n')[:-2]
    encode_dict = {'<pad>':0, 'A': 1, 'C': 2, 'G':3, 'T': 4, '-': 5}
    seq1 = [encode_dict[nuc] for nuc in src]
    seq2 = [encode_dict[nuc] for nuc in tgt]
    return seq1, seq2


def do_pred(iter_,model,device):
    model.eval()
    lst_dfs = []
    with torch.no_grad():
        for i, batch in enumerate(iter_):
            seq1 = batch[3].to(device)
            seq2 = batch[4].to(device)
            out_eff = model(seq1, seq2)
            out_eff = torch.sigmoid(out_eff)
            out_eff = list(out_eff.view(-1).cpu().numpy())
            df_gRNA = pd.DataFrame({'source': batch[0],'target': batch[1]}) 
            df_gRNA['eff_pred'] = out_eff
            lst_dfs.append(df_gRNA)
    df_conc = pd.concat(lst_dfs)
    return df_conc


def get_pred(iter_, model):
    model.eval()
    lst_dfs = []
    with torch.no_grad():
        for i, batch in enumerate(iter_):
            seq1 = batch[3].to(device)
            seq2 = batch[4].to(device)
            y = batch[2].unsqueeze(1).to(device)
            length = batch[5].to(device)

            out_eff = model(seq1, seq2)
            y = list(y.view(-1).cpu().numpy() / 100)
            out_eff = torch.sigmoid(out_eff)
            out_eff = list(out_eff.view(-1).cpu().numpy())

            df_gRNA = pd.DataFrame({'source': batch[0],'target': batch[1], 
                'offtype': batch[-1]})
            df_gRNA['y'] = y
            df_gRNA['y_pred'] = out_eff
            lst_dfs.append(df_gRNA)
    df_conc = pd.concat(lst_dfs)
    return df_conc


def prep_inputs(df_inputs):
    df_inputs['source'] = df_inputs.source.str.upper().str.strip()
    df_inputs['target'] = df_inputs.target.str.upper().str.strip().str.replace('-','')
    df_inputs['target_len'] = df_inputs.target.apply(len)
    df_inputs['efficiency'] = 1
    df_inputs['type'] = 0
    df_inputs.reset_index(drop=True,inplace=True)
    dataset = gRNADataset( df_inputs )
    batches =  DataLoader( dataset, batch_size=5, shuffle=False,
                      collate_fn=generate_batch )
    return batches


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Local version of ABEdeepoff and CBEdeepoff.')
    parser.add_argument('-i', '--input-file', help='Input file include gRNA and offtarget sequences (tab-delimited).')
    parser.add_argument('-o', '--output-file', help='Output table file name.')
    parser.add_argument('-t', '--editor-type', choices=['ABE', 'CBE'], default='ABE', help='Base editor type.')
    args = parser.parse_args()

    if args.editor_type == 'ABE':
        pt_file = 'model/ABEdeepoff.pt'
    else:
        pt_file = 'model/CBEdeepoff.pt'
    
    df = pd.read_csv(args.input_file, sep='\t')
    batches = prep_inputs(df)

    model = Net(ENC_INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, N_LAYERS, ENC_DROPOUT)
    if device == torch.device('cpu'):
        model.load_state_dict(torch.load(pt_file, map_location=device))
    else:
        model.load_state_dict(torch.load(pt_file))
        model.to(device)
    df_eff = do_pred(batches, model, device).reset_index(drop=True)
    
    df_eff.to_csv(args.output_file, sep='\t', index=False, float_format='%.6g')
    

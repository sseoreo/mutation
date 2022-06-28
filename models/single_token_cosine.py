import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy


    
def cosine_similarity_loss(pred, target, margin=0):
    """ target: 0~1
    """
    # print(pred[0], target[0])
    mask = target == 0
    
    loss = 1-pred + torch.clamp(mask * pred - margin, max=0)
    # mask = torch.masked_fill(target, target==0, -1)
    # loss = pred*-1*mask
    # loss = torch.clamp(loss, max=margin)
    # print("after", loss[0])
    
    return loss.mean()

class MutationTokenCosine(nn.Module):
    def __init__(self, args, vocab_size, embedding_dim, hidden_dim):
        super().__init__()

        self.vocab_size =  vocab_size
        self.embedding = Embedding_layer(vocab_size, embedding_dim)
        self.encoder = Encoder(self.embedding, embedding_dim, hidden_dim, hidden_dim, drop_p=args.drop_p)
        # self.classifier_token = nn.Linear(3 * hidden_dim, 1) # last pre_encoer + last post_encoder + temp_hidden ==> 1

        
        
    def forward(self, pre_seq, post_seq):
        """
        :param pre_seq, post_seq: (bsz, len_seq)
        :param trg: (bsz, len_label)
        """

        enc_out, context_out = self.encoder(pre_seq, post_seq)
        
        context_out = context_out.unsqueeze(1).expand(enc_out.size())
        
        out = F.cosine_similarity(enc_out, context_out)

        
        return out


class Embedding_layer(nn.Module):
    def __init__(self, vocab_size, em_dim):
        super().__init__()
        self.size = vocab_size
        self.em_dim = em_dim
        self.embedding_layer = nn.Embedding(self.size, self.em_dim, padding_idx = 0)
        
    def forward(self,x):
        # x = x.cpu().long()
        # print(x, x.shape)
        # x = torch.LongTensor(x)
        return self.embedding_layer(x)

class Encoder(nn.Module):
    def __init__(self, embedding, input_size, enc_hid_dim, dec_hid_dim, num_layer=1, drop_p=0.5):
        super().__init__()
        self.embedding = embedding
        self.input_size = input_size
        self.enc_hid_dim = enc_hid_dim
        self.num_layer = num_layer
        
        self.pre_encoder = nn.GRU(input_size=input_size, hidden_size=enc_hid_dim, num_layers=num_layer, batch_first=True, dropout=drop_p)
        self.post_encoder = nn.GRU(input_size=input_size, hidden_size=enc_hid_dim, num_layers=num_layer, batch_first=True, dropout=drop_p)
        self.dropout = nn.Dropout(drop_p)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

    def forward(self, prefix, postfix):
        """
        :param prefix: (len_seq, bsz)
        :param postfix: (len_seq, bsz)
        :return stacked_out: (len_sze, bsz, 2* hidden_dim)
        :return (stacked_hidden, stacked_cell): (n_layer, bsz, 2* hidden_dim)
        """
        assert prefix.size(1) == postfix.size(1)


        pre_batch = self.dropout(self.embedding(prefix))
        post_batch = self.dropout(self.embedding(postfix))
        # print(post_batch.shape)
        pre_out, pre_hidden = self.pre_encoder(pre_batch)
        post_out, post_hidden = self.post_encoder(post_batch)
        # print(pre_out.shape, pre_hidden.shape, pre_cell.shape)
        
        # (len_pre+len_post), bsz, hid_dim
        stacked_out = torch.cat((pre_out, post_out), dim = 1)
        
        # bsz, hid_dim 
        context_out = self.fc(torch.cat((pre_out[:, -1, :], post_out[:, -1, :]), dim = -1))
        # print("preout", pre_out.shape, context_out.shape)
        
        return stacked_out, context_out


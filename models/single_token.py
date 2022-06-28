import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy


class MutationToken(nn.Module):
    def __init__(self,vocab_size,em_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = Embedding_layer(vocab_size, em_dim)
        self.pre_encoder = PreEncoder(self.embedding, em_dim, hidden_size = hidden_dim)
        self.post_encoder = PostEncoder(self.embedding, em_dim, hidden_size = hidden_dim)
        # self.classifier = nn.Linear(2 * hidden_dim, vocab_size)
        
        self.classifier_token = nn.Linear(3 * hidden_dim, 1) # last pre_encoer + last post_encoder + temp_hidden ==> 1
        
        
    def forward(self, pre_seq, post_seq):
        
        # print(pre_seq.shape, post_seq.shape)
        bsz, pre_len = pre_seq.shape
        post_len = post_seq.size(-1)
        pre_out, pre_out_last = self.pre_encoder(pre_seq)
        post_out, post_out_last = self.post_encoder(post_seq)
        # print(pre_out.shape, post_out.shape, pre_out_last.shape, post_out_last.shape)

        concatenated_enc = torch.cat((pre_out, post_out), dim = 1) # bsz, pre_len+post_len, hidden_dim
        pre_out_last = pre_out_last.unsqueeze(1).expand(concatenated_enc.size())
        post_out_last = post_out_last.unsqueeze(1).expand(concatenated_enc.size())
        out = torch.cat([concatenated_enc, pre_out_last, post_out_last], dim=-1) # bsz, pre_len+post_len, 3*hidden_dim
        
        out = self.classifier_token(F.relu(out)).cuda() # batch_size * pre_len+post_len * 1

        return torch.sigmoid(out)


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
    
class PreEncoder(nn.Module):
    def __init__(self, embedding, input_size, num_layer = 1, hidden_size = 128, mode_lm = False):
        super().__init__()
        self.embedding = embedding
        
        self.mode_lm = mode_lm
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        
        
        self.encoder = nn.LSTM(input_size = input_size, hidden_size = self.hidden_size, num_layers =self.num_layer,batch_first = True)
        
        
        
    def init_states(self, b):
        self.h = torch.zeros(self.num_layer, b, self.hidden_size).cuda()
        self.c = torch.zeros(self.num_layer, b, self.hidden_size).cuda()
        
    def forward(self,x):
        b = x.size(0)
        self.init_states(b)
        embedded_batch = self.embedding(x)
        
        output,_ = self.encoder(embedded_batch, (self.h, self.c))
        # print(output.shape)
        last_output = output[:, -1, :]
        
        return output, last_output
        
        
class PostEncoder(nn.Module):
    def __init__(self, embedding, input_size, num_layer = 1, hidden_size = 128, mode_lm = False):
        super().__init__()
        self.embedding = embedding
        self.mode_lm = mode_lm
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        
        
        self.encoder = nn.LSTM(input_size = input_size, hidden_size = self.hidden_size, num_layers =self.num_layer,batch_first = True)
        
        
        
    def init_states(self, b):
        self.h = torch.zeros(self.num_layer, b, self.hidden_size).cuda()
        self.c = torch.zeros(self.num_layer, b, self.hidden_size).cuda()
        
    def forward(self,x):
        b = x.size(0)
        self.init_states(b)
        embedded_batch = self.embedding(x)
        
        output, _ = self.encoder(embedded_batch, (self.h, self.c))
        last_output = output[:, -1, :]
        
        return output, last_output       
        
        
    
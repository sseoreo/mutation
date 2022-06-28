import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import random

class Seq2SeqType(nn.Module):
    def __init__(self, args, vocab_size, embedding_dim, hidden_dim, mode):
        super().__init__()

        self.vocab_size =  vocab_size
        self.embedding = Embedding_layer(vocab_size, embedding_dim)
        self.encoder = Encoder(self.embedding, embedding_dim, hidden_dim, hidden_dim, drop_p=args.drop_p)
        self.decoder = Decoder(self.embedding, vocab_size, embedding_dim, hidden_dim, drop_p=args.drop_p)

        self.fc_hidden = nn.Linear(hidden_dim*2, embedding_dim)
        self.fc_cell = nn.Linear(hidden_dim*2, embedding_dim)

        
        
    def forward(self, pre_seq, post_seq, trg, teacher_forcing_ratio = 0.5):
        """
        :param pre_seq, post_seq: (bsz, len_seq)
        :param trg: (bsz, len_label)
        """
        # print("model input", pre_seq.shape, post_seq.shape, trg.shape)

        pre_seq, post_seq, trg = pre_seq.transpose(1, 0), post_seq.transpose(1, 0), trg.transpose(1, 0)
        # print("model input (transpose)", pre_seq.shape, post_seq.shape, trg.shape)

        max_len, bsz = trg.shape
        outputs = torch.zeros(max_len, bsz, self.vocab_size).cuda()

        # enc_out: (bsz, seq_len, 2*hidden_dim), prev_hidden: (n_layer, bsz, 2*hidden_dim) 
        enc_out, (prev_hidden, prev_cell) = self.encoder(pre_seq, post_seq)
        

        # first input. 
        # input = trg[0, :]
        input = pre_seq[-1, :]

        for t in range(0, max_len):

            # output: (bsz, vocab)
            output, (prev_hidden, prev_cell) = self.decoder(input, prev_hidden, prev_cell, enc_out)

            # outputs: (max_len, bsz, vocab)
            outputs[t] = output
            
            teacher_force = random.random() < teacher_forcing_ratio
            pred = output.argmax(-1)
            
            # print(pred.shape, pred[0], trg[t-1][0])
            input = trg[t] if teacher_force else pred
            
        # print(outputs.argmax(-1)[:, 0], trg[:, 0])
        return outputs.transpose(1, 0)


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
        
        self.pre_encoder = nn.LSTM(input_size=input_size, hidden_size=self.enc_hid_dim, num_layers=self.num_layer, batch_first=False, dropout=drop_p)
        self.post_encoder = nn.LSTM(input_size=input_size, hidden_size=self.enc_hid_dim, num_layers=self.num_layer, batch_first=False, dropout=drop_p)
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
        pre_out, (pre_hidden, pre_cell) = self.pre_encoder(pre_batch)
        post_out, (post_hidden, post_cell) = self.post_encoder(post_batch)
        # print(pre_out.shape, pre_hidden.shape, pre_cell.shape)
        
        # (len_pre+len_post), bsz, hid_dim
        stacked_out = torch.cat((pre_out, post_out), dim = 0) 
        
        # bsz, len_seq, 2 * hidden_dim
        stacked_hidden = torch.tanh(self.fc(torch.cat((pre_hidden, post_hidden), dim = -1)))
        stacked_cell = torch.tanh(self.fc(torch.cat((pre_cell, post_cell), dim = -1)))
        
        return stacked_out, (stacked_hidden, stacked_cell)


        
class Decoder(nn.Module):
    def __init__(self, embedding, num_embddings, embedding_dim, dec_hid_dim, num_layer=1, drop_p=0.5):
        super().__init__()
        self.embedding = embedding

        self.rnn = nn.LSTM(embedding_dim, dec_hid_dim, num_layers=num_layer, batch_first=False, dropout=drop_p)
        self.fc_out = nn.Linear(dec_hid_dim, num_embddings)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, input, prev_hidden, prev_cell, *args):
        """
        :param input: (1, bsz)
        :param prev_hidden: (n_layer, bsz, dec_hid_dim)
        :param prev_cell: (n_layer, bsz, dec_hid_dim)
        :return output: (1, bsz, vocab_size)
        """
        # bsz = input.size(0)
        input = input.unsqueeze(0) # (1, bsz) for rnn
        embed = self.dropout(self.embedding(input))
 

        # output: (1, bsz, hidden_dim)
        output, (hidden, cell) = self.rnn(embed, (prev_hidden, prev_cell))
        # print(output.shape, embed.shape, prev_hidden.shape, prev_cell.shape)
        
        output = self.fc_out(output.squeeze(0))
        return output, (hidden, cell)

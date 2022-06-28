import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import random

class MutationSeq2SeqAttn(nn.Module):
    def __init__(self, args, vocab_size, embedding_dim, hidden_dim):
        super().__init__()

        self.vocab_size =  vocab_size
        self.embedding = Embedding_layer(vocab_size, embedding_dim)
        

        self.encoder = Encoder(self.embedding, embedding_dim, hidden_dim, hidden_dim, drop_p=args.drop_p)
        self.attention = Attention(hidden_dim, hidden_dim)
        self.decoder = Decoder(self.embedding, vocab_size, embedding_dim, hidden_dim, hidden_dim, self.attention, drop_p=args.drop_p)

        # self.fc_hidden = nn.Linear(hidden_dim*2, embedding_dim)
        # self.fc_cell = nn.Linear(hidden_dim*2, embedding_dim)

        
        
    def forward(self, pre_seq, post_seq, trg, teacher_forcing_ratio = 0.5):
        """
        :param pre_seq, post_seq: (bsz, len_seq)
        :param trg: (bsz, len_label)
        """
        # print("model input", pre_seq.shape, post_seq.shape, trg.shape)

        pre_seq, post_seq, trg = pre_seq.transpose(1, 0), post_seq.transpose(1, 0), trg.transpose(1, 0)

        max_len, bsz = trg.shape
        outputs = torch.zeros(max_len, bsz, self.vocab_size).cuda()

        # enc_out: (bsz, seq_len, 2*hidden_dim), prev_hidden: (n_layer, bsz, 2*hidden_dim) 
        enc_out, (prev_hidden, prev_cell) = self.encoder(pre_seq, post_seq)
        # prev_hidden = self.fc_hidden(prev_hidden) 
        # prev_cell = self.fc_hidden(prev_cell)

        # first input. 
        input = trg[0, :]
        # input = pre_seq[-1, :]

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
        :return (stacked_hidden, stacked_cell): (bsz, 2* hidden_dim)
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
        
        # return only last layer
        # bsz, 2 * hidden_dim
        stacked_hidden = torch.tanh(self.fc(torch.cat((pre_hidden, post_hidden), dim = -1))[-1])
        stacked_cell = torch.tanh(self.fc(torch.cat((pre_cell, post_cell), dim = -1))[-1])
        
        return stacked_out, (stacked_hidden, stacked_cell)


        
class Decoder(nn.Module):
    def __init__(self, embedding, num_embddings, embedding_dim, enc_hid_dim, dec_hid_dim, attention, drop_p=0.5, ):
        super().__init__()
        self.embedding = embedding
        self.attention = attention

        self.rnn = nn.LSTM( enc_hid_dim +embedding_dim, dec_hid_dim, batch_first=False, dropout=drop_p)
        self.fc_out = nn.Linear(enc_hid_dim +embedding_dim+dec_hid_dim, num_embddings)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, input, prev_hidden, prev_cell, enc_out):
        """
        :param input: (1, bsz)
        :param prev_hidden: (bsz, hidden_dim*2)
        :param prev_cell: (bsz, hidden_dim*2)
        :return output: (1, bsz, vocab_size)
        """
        # bsz = input.size(0)
        input = input.unsqueeze(0) # (1, bsz) for rnn
        embed = self.dropout(self.embedding(input))

        # bsz, 1, src_len
        attn = self.attention(prev_hidden, enc_out).unsqueeze(1)
        
        # bsz, src_len, 2*enc_hid_dim
        enc_out = enc_out.permute(1, 0, 2)

        # batch-wise mat. multiplication
        # bsz, 1, 2*enc_hid_dim
        weighted = torch.bmm(attn, enc_out)

        # 1, bsz, 2*enc_hid_dim
        weighted = weighted.permute(1, 0, 2)

        # 1, bsz, emb_dim+2*enc_hid_dim
        rnn_input = torch.cat((embed, weighted), dim = 2)

        # output: (1, bsz, hidden_dim)
        # print(rnn_input.shape, embed.shape, weighted.shape, prev_hidden.shape, prev_cell.shape)
        output, (hidden, cell) = self.rnn(rnn_input, (prev_hidden.unsqueeze(0), prev_cell.unsqueeze(0)))
        
        # print(output.shape, embed.shape, prev_hidden.shape, prev_cell.shape)
        assert (output == hidden).all()

        embed = embed.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        # bsz, emb_dim + 2*enc_hid_dim + dec_hid_dim
        output = torch.cat((output, weighted, embed), dim = 1)

        output = self.fc_out(output)

        return output, (hidden.squeeze(0), cell.squeeze(0))

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim) -> None:
        super().__init__()
            
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)

    def forward(self, hidden, enc_out):
        """
        :param hidden: (1, bsz, dec_hid_dim) 
        :param enc_out: (src_len, bsz, enc_hid_dim)
        :return attn: (bsz, src_len)
        """
        src_len, bsz, _ = enc_out.shape
        
        # bsz, src_len, dec_hid_dim
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # src_len, bsz, enc_hid_dim
        enc_out = enc_out.permute(1, 0, 2)

        # bsz, src_len, dec_hid_dim
        energy = torch.tanh(self.attn(torch.cat([hidden, enc_out], dim=2)))

        # bsz, src_len
        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)
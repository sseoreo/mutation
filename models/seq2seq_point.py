import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import random

class Seq2SeqPoint(nn.Module):
    def __init__(self, args, vocab_size, embedding_dim, hidden_dim, mode):
        super().__init__()

        self.vocab_size =  vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = Encoder(self.embedding, embedding_dim, hidden_dim, hidden_dim, drop_p=args.drop_p)
        self.decoder = Decoder(1, hidden_dim, hidden_dim, drop_p=args.drop_p)
        
        # self.decoder = Decoder(1, hidden_dim, hidden_dim, self.attn)
        
        self.fc_hidden = nn.Linear(hidden_dim*2, embedding_dim)
        # self.fc_cell = nn.Linear(hidden_dim*2, embedding_dim)

        
        
    def forward(self, pre_seq, post_seq, trg, teacher_forcing_ratio = 0.5):
        """
        :param pre_seq, post_seq: (bsz, len_seq)
        :param trg: (bsz, len_label)
        """
        # print("model input", pre_seq.shape, post_seq.shape, trg.shape)


        bsz, max_len, = trg.shape
        assert pre_seq.shape == post_seq.shape
        bsz, len_pre = pre_seq.shape 

        outputs_pre = torch.zeros(bsz, max_len, len_pre, 1).cuda()
        outputs_post = torch.zeros(bsz, max_len, len_pre, 1).cuda()

        # enc_out: (bsz, seq_len, 2*hidden_dim), prev_hidden: (n_layer, bsz, 2*hidden_dim) 
        enc_out, prev_hidden = self.encoder(pre_seq, post_seq)
        # prev_hidden = self.fc_hidden(prev_hidden) 
        # prev_cell = self.fc_hidden(prev_cell)

        # first input. 
        # input = trg[0, :]
        # input = pre_seq[:, -1]

        for t in range(0, max_len):

            # output: (bsz, vocab)
            output, prev_hidden = self.decoder(prev_hidden, enc_out)

            # outputs: (bsz, max_len, vocab)
            # print(outputs_pre[:, t, :].shape, output[:, :len_pre, :].shape)
            outputs_pre[:, t, :, :] = torch.sigmoid(output[:, :len_pre, :])
            outputs_post[:, t, :, :] = torch.sigmoid(output[:, len_pre:, :])
            
        # print(outputs.argmax(-1)[:, 0], trg[:, 0])
        return outputs_pre, outputs_post


class Seq2SeqPointAttn(nn.Module):
    def __init__(self, args, vocab_size, embedding_dim, hidden_dim, mode):
        super().__init__()

        self.vocab_size =  vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.encoder = Encoder(self.embedding, embedding_dim, hidden_dim, hidden_dim, drop_p=args.drop_p)
        self.attention = Attention(hidden_dim, hidden_dim)
        self.decoder = DecoderAttn(1, hidden_dim, hidden_dim, self.attention, drop_p=args.drop_p)

        # self.fc_hidden = nn.Linear(hidden_dim*2, embedding_dim)
        # self.fc_cell = nn.Linear(hidden_dim*2, embedding_dim)

        
        
    def forward(self, pre_seq, post_seq, trg, teacher_forcing_ratio = 0.5):
        """
        :param pre_seq, post_seq: (bsz, len_seq)
        :param trg: (bsz, len_label)
        """
        # print("model input", pre_seq.shape, post_seq.shape, trg.shape)


        bsz, max_len, = trg.shape
        assert pre_seq.shape == post_seq.shape
        bsz, len_pre = pre_seq.shape 

        outputs_pre = torch.zeros(bsz, max_len, len_pre, 1).cuda()
        outputs_post = torch.zeros(bsz, max_len, len_pre, 1).cuda()

        # enc_out: (bsz, seq_len, 2*hidden_dim), prev_hidden: (n_layer, bsz, 2*hidden_dim) 
        enc_out, prev_hidden = self.encoder(pre_seq, post_seq)
        # prev_hidden = self.fc_hidden(prev_hidden) 
        # prev_cell = self.fc_hidden(prev_cell)

        # first input. 
        # input = trg[0, :]
        # input = pre_seq[:, -1]

        for t in range(0, max_len):

            # output: (bsz, vocab)
            output, prev_hidden = self.decoder(prev_hidden, enc_out)

            # outputs: (bsz, max_len, vocab)
            # print(outputs_pre[:, t, :].shape, output[:, :len_pre, :].shape)
            outputs_pre[:, t, :, :] = torch.sigmoid(output[:, :len_pre, :])
            outputs_post[:, t, :, :] = torch.sigmoid(output[:, len_pre:, :])
            
        # print(outputs.argmax(-1)[:, 0], trg[:, 0])
        return outputs_pre, outputs_post

class Encoder(nn.Module):
    def __init__(self, embedding, emb_dim, enc_hid_dim, dec_hid_dim, num_layers=1, drop_p=0.5, batch_first=True):
        super().__init__()
        self.embedding = embedding
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        
        
        self.pre_enc_layer = nn.GRU(input_size=emb_dim, hidden_size=self.enc_hid_dim, num_layers=num_layers, dropout=drop_p, batch_first=batch_first)
        self.post_enc_layer = nn.GRU(input_size=emb_dim, hidden_size=self.enc_hid_dim, num_layers=num_layers, dropout=drop_p, batch_first=batch_first)
        self.dropout = nn.Dropout(drop_p)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

    def forward(self, pre_seq, post_seq):
        """
        :param prefix: (bsz, len_pre)
        :param postfix: (bsz, len_post)
        :return enc_out: (bsz, len_pre+len_post, enc_hid_dim)
        :return enc_hidden: (bsz, dec_hid_dim)
        """
        assert pre_seq.size(1) == post_seq.size(1)

        pre_seq = self.dropout(self.embedding(pre_seq))
        post_seq = self.dropout(self.embedding(post_seq)) 
        
        pre_out, pre_hidden = self.pre_enc_layer(pre_seq)
        post_out, post_hidden = self.post_enc_layer(post_seq)
        # print(pre_out.shape, pre_hidden.shape)
        
        # bsz, (len_pre+len_post), hid_dim
        enc_out = torch.cat((pre_out, post_out), dim = 1) 
        
        # return only last layer
        enc_hidden = torch.tanh(self.fc(torch.cat((pre_hidden[-1, :, :], post_hidden[-1, :, :]), dim = -1)))
        
        return enc_out, enc_hidden


class Decoder(nn.Module):
    def __init__(self, n_classes, enc_hid_dim, dec_hid_dim, num_layer=1, drop_p=0.5):
        super().__init__()
        # self.embedding = embedding

        # self.rnn = nn.GRU(embedding_dim, dec_hid_dim, num_layers=num_layer, batch_first=True, dropout=drop_p)
        self.fc1 = nn.Linear(dec_hid_dim, dec_hid_dim)
        
        self.fc_out = nn.Linear(2*dec_hid_dim, n_classes)
        # self.dropout = nn.Dropout(drop_p)

    def forward(self, hidden, enc_out):
        """
        :param input: (1, bsz)
        :param prev_hidden: (n_layer, bsz, dec_hid_dim)
        :param prev_cell: (n_layer, bsz, dec_hid_dim)
        :return output: (1, bsz, vocab_size)
        """
        # bsz = input.size(0)
        # input = input.unsqueeze(1) # (1, bsz) for rnn
        # embed = self.dropout(self.embedding(input))

        hidden = torch.tanh(self.fc1(hidden))

        output = hidden.unsqueeze(1).repeat(1, enc_out.size(1), 1)
        
        output = F.relu(torch.cat([enc_out, output], dim=-1))


        # print(embed.shape, prev_hidden.shape)
        # output: (bsz, 1,  hidden_dim)
        
        
        output = self.fc_out(output)
        # print(output.shape, hidden.shape)
        
        return output, hidden
        
class DecoderAttn(nn.Module):
    def __init__(self, n_classes, enc_hid_dim, dec_hid_dim, attention, drop_p=0.5):
        super().__init__()
        
        self.dec_layer = nn.GRU(input_size=dec_hid_dim, hidden_size=dec_hid_dim, num_layers=1, dropout=drop_p, batch_first=True)
        
        self.fc1 = nn.Linear(dec_hid_dim, dec_hid_dim)
        self.fc2 = nn.Linear(dec_hid_dim+enc_hid_dim, dec_hid_dim)

        self.fc_out = nn.Linear(dec_hid_dim+enc_hid_dim, n_classes)
        self.attention = attention

    def forward(self, hidden, encoder_outputs):

        # [batch size, src len]
        a = self.attention(hidden, encoder_outputs)
                
        # [batch size, 1, src len]
        a = a.unsqueeze(1)
        
        
        # [batch size, 1, enc hid dim]
        weighted = torch.bmm(a, encoder_outputs).squeeze(1)
        

        # [batch size, enc hid dim]
        output = torch.tanh(self.fc1(hidden))
        # print(output.shape, weighted.shape)

        # [batch size, dec hid dim]
        output = torch.tanh(self.fc2(torch.cat((output, weighted), dim = 1)))
        
        # [batch size, len_pre+len_post, dec_hid_dim]
        output = output.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)

        # [batch size, len_pre+len_post, dec_hid_dim+enc_hid_dim]
        output = F.relu(torch.cat([encoder_outputs, output], dim=-1))
        

        # enc_hidden: (bsz, len_pre+len_post, 1)
        output = self.fc_out(output)
        
        return output, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):

        #hidden = [batch size, dec_hid_dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)
        
        #repeat decoder hidden state src_len times
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        # print(hidden.shape, encoder_outputs.shape, torch.cat((hidden, encoder_outputs), dim = -1).shape)
        
        # [batch size, src len, dec hid dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = -1))) 
        
        # print(hidden.shape, encoder_outputs.shape)
        
        attention = self.v(energy).squeeze(2)
        
        #attention= [batch size, src len]
        return F.softmax(attention, dim=1)
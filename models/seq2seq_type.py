import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import random

class Seq2SeqType(nn.Module):
    def __init__(self, args, vocab_size, embedding_dim, hidden_dim, mode):
        super().__init__()

        self.vocab_size =  vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = Encoder(self.embedding, embedding_dim, hidden_dim, hidden_dim, drop_p=args.drop_p)
        self.decoder = Decoder(self.embedding, vocab_size, embedding_dim, hidden_dim, drop_p=args.drop_p)

        
        
    def forward(self, pre_seq, post_seq, trg, teacher_forcing_ratio = 0.5):
        """
        :param pre_seq, post_seq: (bsz, src_len)
        :param trg: (bsz, max_len)
        :return outputs: (bsz, max_len, n_vocab)
        """

        bsz, max_len, = trg.shape
        outputs = torch.zeros(bsz, max_len, self.vocab_size).cuda()

        # enc_out: (bsz, enc_len, enc_hid_dim)
        # prev_hidden: (n_layer, bsz, dec_hid_dim) 
        enc_out, prev_hidden = self.encoder(pre_seq, post_seq)
        
        # first input. 
        # input = trg[0, :]
        input = pre_seq[:, -1]

        for t in range(0, max_len):

            # output: (bsz, vocab_size)
            output, prev_hidden = self.decoder(input, prev_hidden, enc_out)

            # outputs: (bsz, max_len, vocab_size)
            outputs[:, t] = output
            
            teacher_force = random.random() < teacher_forcing_ratio
            
            # outputs: (bsz, max_len)
            pred = output.argmax(-1)
            
            input = trg[:, t] if teacher_force else pred
            
        return outputs


class Seq2SeqTypeAttn(nn.Module):
    def __init__(self, args, vocab_size, embedding_dim, hidden_dim, mode):
        super().__init__()

        self.vocab_size =  vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.encoder = Encoder(self.embedding, embedding_dim, hidden_dim, hidden_dim, drop_p=args.drop_p)
        self.attention = Attention(hidden_dim, hidden_dim)
        self.decoder = DecoderAttn(self.embedding, vocab_size, embedding_dim, hidden_dim, hidden_dim, self.attention, drop_p=args.drop_p)

        
    def forward(self, pre_seq, post_seq, trg, teacher_forcing_ratio = 0.5):
        """
        :param pre_seq, post_seq: (bsz, src_len)
        :param trg: (bsz, max_len)
        :return outputs: (bsz, max_len, n_vocab)
        """

        bsz, max_len, = trg.shape
        outputs = torch.zeros(bsz, max_len, self.vocab_size).cuda()


        # enc_out: (bsz, enc_len, enc_hid_dim)
        # prev_hidden: (n_layer, bsz, dec_hid_dim) 
        enc_out, prev_hidden = self.encoder(pre_seq, post_seq)
        
        # first input. 
        # input = trg[0, :]
        input = pre_seq[:, -1]

        for t in range(0, max_len):

            # output: (bsz, vocab_size)
            output, prev_hidden = self.decoder(input, prev_hidden, enc_out)

            # outputs: (bsz, max_len, vocab_size)
            outputs[:, t] = output
            
            teacher_force = random.random() < teacher_forcing_ratio
            
            # outputs: (bsz, max_len)
            pred = output.argmax(-1)
            
            
            input = trg[:, t] if teacher_force else pred
            
        return outputs


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
        :return enc_out: (bsz, enc_len, enc_hid_dim)
        :return enc_hidden: (bsz, dec_hid_dim)
        """
        assert pre_seq.size(1) == post_seq.size(1)

        pre_seq = self.dropout(self.embedding(pre_seq))
        post_seq = self.dropout(self.embedding(post_seq)) 
        
        pre_out, pre_hidden = self.pre_enc_layer(pre_seq)
        post_out, post_hidden = self.post_enc_layer(post_seq)
        
        # bsz, (len_pre+len_post), enc_hid_dim
        enc_out = torch.cat((pre_out, post_out), dim = 1) 
        
        # return only last layer
        enc_hidden = torch.tanh(self.fc(torch.cat((pre_hidden[-1, :, :], post_hidden[-1, :, :]), dim = -1)))
        
        return enc_out, enc_hidden


        
class Decoder(nn.Module):
    def __init__(self, embedding, num_embddings, embedding_dim, dec_hid_dim, num_layer=1, drop_p=0.5):
        super().__init__()
        self.embedding = embedding

        self.rnn = nn.GRU(embedding_dim, dec_hid_dim, num_layers=num_layer, batch_first=True, dropout=drop_p)
        self.fc_out = nn.Linear(dec_hid_dim, num_embddings)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, input, prev_hidden, *args):
        """
        :param input: (bsz)
        :param prev_hidden: (bsz, dec_hid_dim)
        :param enc_out: (bsz, enc_len, enc_hid_dim)
        :return output: (bsz, 1, vocab_size)
        """
        # bsz = input.size(0)
        input = input.unsqueeze(1) # (bsz, 1) for rnn
        embed = self.dropout(self.embedding(input))
 
        # bsz, 1, dec_hid_dim
        output, hidden = self.rnn(embed, prev_hidden.unsqueeze(0))
        
        # bsz, 1, vocab_size
        output = self.fc_out(output.squeeze(1))

        return output, hidden.squeeze(0)


        

class DecoderAttn(nn.Module):
    def __init__(self, embedding, num_embddings, embedding_dim, enc_hid_dim, dec_hid_dim, attention, drop_p=0.5, ):
        super().__init__()
        self.embedding = embedding
        self.attention = attention

        self.rnn = nn.GRU( enc_hid_dim +embedding_dim, dec_hid_dim, batch_first=True, dropout=drop_p)
        self.fc_out = nn.Linear(enc_hid_dim +embedding_dim+dec_hid_dim, num_embddings)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, input, prev_hidden, enc_out):
        """
        :param input: (bsz)
        :param prev_hidden: (bsz, dec_hid_dim)
        :param enc_out: (bsz, enc_len, enc_hid_dim)
        :return output: (bsz, 1, vocab_size)
        """
        # bsz = input.size(0)
        input = input.unsqueeze(1) # (bsz, 1) for rnn
        embed = self.dropout(self.embedding(input))

        # bsz, enc_len
        a = self.attention(prev_hidden, enc_out)
        
        # bsz, 1, enc_len
        a = a.unsqueeze(1)
        
        # bsz, 1, enc_hid_dim
        weighted = torch.bmm(a, enc_out)

        # bsz, 1, emb_dim+enc_hid_dim
        rnn_input = torch.cat((embed, weighted), dim = -1)

        # output: (bsz, 1, dec_hid_dim)
        output, hidden = self.rnn(rnn_input, prev_hidden.unsqueeze(0))
        

        embed = embed.squeeze(1)
        output = output.squeeze(1)
        weighted = weighted.squeeze(1)
        
        
        # bsz, emb_dim + enc_hid_dim + dec_hid_dim
        output = torch.cat((output, weighted, embed), dim = 1)
        
        # bsz, vocab_size
        output = self.fc_out(output)

        return output, hidden.squeeze(0)


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        """
        :param hidden: (bsz, dec_hid_dim)
        :param encoder_outputs: (bsz, enc_len, enc_hid_dim)
        :return attention: (bsz, enc_len) 
        """
        
        batch_size = encoder_outputs.size(0)
        enc_len = encoder_outputs.size(1)
        
        # bsz, enc_len, dec_hid_dim
        hidden = hidden.unsqueeze(1).repeat(1, enc_len, 1)
        
        # bsz, enc_len, dec_hid_dim
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = -1))) 
        
        
        # bsz, enc_len
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim=1)
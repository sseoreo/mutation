import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import random

class Seq2SeqAll(nn.Module):
    def __init__(self, args, vocab_size, embedding_dim, hidden_dim, mode):
        super().__init__()

        self.vocab_size =  vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = Encoder(self.embedding, embedding_dim, hidden_dim, hidden_dim, drop_p=args.drop_p)
        # self.decoder = Decoder(1, hidden_dim, hidden_dim, drop_p=args.drop_p)
        self.decoder = Decoder(self.embedding, vocab_size, embedding_dim, hidden_dim, drop_p=args.drop_p)


    def forward(self, pre_seq, post_seq, trg, teacher_forcing_ratio = 0.5):
        """
        :param pre_seq, post_seq: (bsz, len_seq)
        :param trg: (bsz, len_label)
        """
        # print("model input", pre_seq.shape, post_seq.shape, trg.shape)


        bsz, max_len, = trg.shape
        assert pre_seq.shape == post_seq.shape
        bsz, len_pre = pre_seq.shape 

        outputs_type =  torch.zeros(bsz, max_len, self.vocab_size).cuda()
        outputs_pre = torch.zeros(bsz, max_len, len_pre, 1).cuda()
        outputs_post = torch.zeros(bsz, max_len, len_pre, 1).cuda()

        # enc_out: (bsz, seq_len, 2*hidden_dim), prev_hidden: (n_layer, bsz, 2*hidden_dim) 
        enc_out, prev_hidden = self.encoder(pre_seq, post_seq)
        # prev_hidden = self.fc_hidden(prev_hidden) 
        # prev_cell = self.fc_hidden(prev_cell)

        # first input. 
        input = pre_seq[:, -1]

        for t in range(0, max_len):

            # output: (bsz, vocab)
            out_type, out_point, prev_hidden = self.decoder(input, prev_hidden, enc_out)

            # outputs: (bsz, max_len, vocab)
            # print(outputs_pre[:, t, :].shape, output[:, :len_pre, :].shape)
            outputs_type[:, t] = out_type
            outputs_pre[:, t, :, :] = torch.sigmoid(out_point[:, :len_pre, :])
            outputs_post[:, t, :, :] = torch.sigmoid(out_point[:, len_pre:, :])

            teacher_force = random.random() < teacher_forcing_ratio
            pred = out_type.argmax(-1)
            
            input = trg[:, t] if teacher_force else pred

        # print(outputs.argmax(-1)[:, 0], trg[:, 0])
        return outputs_type, outputs_pre, outputs_post


class Seq2SeqAllAttn(nn.Module):
    def __init__(self, args, vocab_size, embedding_dim, hidden_dim, mode):
        super().__init__()

        self.vocab_size =  vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.encoder = Encoder(self.embedding, embedding_dim, hidden_dim, hidden_dim, drop_p=args.drop_p)
        self.attention = Attention(hidden_dim, hidden_dim)
        # self.decoder = DecoderAttn(1, hidden_dim, hidden_dim, self.attention, drop_p=args.drop_p)
        self.decoder = DecoderAttn(self.embedding, vocab_size, embedding_dim, hidden_dim, hidden_dim, self.attention, drop_p=args.drop_p)

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

        outputs_type =  torch.zeros(bsz, max_len, self.vocab_size).cuda()
        outputs_pre = torch.zeros(bsz, max_len, len_pre, 1).cuda()
        outputs_post = torch.zeros(bsz, max_len, len_pre, 1).cuda()

        # enc_out: (bsz, seq_len, 2*hidden_dim), prev_hidden: (n_layer, bsz, 2*hidden_dim) 
        enc_out, prev_hidden = self.encoder(pre_seq, post_seq)
        # prev_hidden = self.fc_hidden(prev_hidden) 
        # prev_cell = self.fc_hidden(prev_cell)

        # first input. 
        input = pre_seq[:, -1]

        for t in range(0, max_len):

            # output: (bsz, vocab)
            out_type, out_point, prev_hidden = self.decoder(input, prev_hidden, enc_out)

            # outputs: (bsz, max_len, vocab)
            # print(outputs_pre[:, t, :].shape, output[:, :len_pre, :].shape)
            outputs_type[:, t] = out_type
            outputs_pre[:, t, :, :] = torch.sigmoid(out_point[:, :len_pre, :])
            outputs_post[:, t, :, :] = torch.sigmoid(out_point[:, len_pre:, :])

            teacher_force = random.random() < teacher_forcing_ratio
            pred = out_type.argmax(-1)
            
            input = trg[:, t] if teacher_force else pred

        # print(outputs.argmax(-1)[:, 0], trg[:, 0])
        return outputs_type, outputs_pre, outputs_post

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
    def __init__(self, embedding, num_embddings, embedding_dim, dec_hid_dim, num_layer=1, drop_p=0.5):
        super().__init__()
        self.embedding = embedding

        self.rnn = nn.GRU(embedding_dim, dec_hid_dim, num_layers=num_layer, batch_first=True, dropout=drop_p)
        self.fc_type = nn.Linear(dec_hid_dim, num_embddings)
        self.fc_point = nn.Linear(2*dec_hid_dim, 1)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, input, prev_hidden, enc_out):
        """
        :param input: (1, bsz)
        :param prev_hidden: (n_layer, bsz, dec_hid_dim)
        :param prev_cell: (n_layer, bsz, dec_hid_dim)
        :return output: (1, bsz, vocab_size)
        """
        # bsz = input.size(0)
        input = input.unsqueeze(1) # (1, bsz) for rnn
        
        embed = self.dropout(self.embedding(input))
        
 
        # print(embed.shape, prev_hidden.shape)
        # output: (bsz, 1,  hidden_dim)
        output, hidden = self.rnn(embed, prev_hidden.unsqueeze(0))

        print(output.shape, embed.shape, enc_out.shape)
        
        out_point = output.repeat(1, enc_out.size(1), 1)
        
        out_point = F.relu(torch.cat([enc_out, out_point], dim=-1))

        out_point = self.fc_point(out_point)
        out_type = self.fc_type(output.squeeze(1))
        return out_type, out_point, hidden


class DecoderAttn(nn.Module):
    def __init__(self, embedding, num_embddings, embedding_dim, enc_hid_dim, dec_hid_dim, attention, num_layer=1, drop_p=0.5):
        super().__init__()

        self.embedding = embedding
        self.attention = attention

        # self.rnn = nn.GRU( enc_hid_dim +embedding_dim, dec_hid_dim, batch_first=True, dropout=drop_p)
        self.rnn = nn.GRU(enc_hid_dim + embedding_dim, dec_hid_dim, num_layers=num_layer, batch_first=True, dropout=drop_p)
        
        self.fc_type = nn.Linear(dec_hid_dim+enc_hid_dim+embedding_dim, num_embddings)
        self.fc_point = nn.Linear(2*dec_hid_dim+enc_hid_dim+embedding_dim, 1)
        self.dropout = nn.Dropout(drop_p)
    
    def forward(self, input, prev_hidden, enc_out):
        """
        :param input: (1, bsz)
        :param prev_hidden: (bsz, hidden_dim*2)
        :param prev_cell: (bsz, hidden_dim*2)
        :return output: (1, bsz, vocab_size)
        """
        # bsz = input.size(0)
        input = input.unsqueeze(1) # (bsz, 1) for rnn
        embed = self.dropout(self.embedding(input))

        # bsz, src_len
        a = self.attention(prev_hidden, enc_out)
        
        # bsz, 1, src_len
        a = a.unsqueeze(1)
        
        # bsz, 1, enc hid dim
        weighted = torch.bmm(a, enc_out)
        # print(embed.shape, weighted.shape)

        # bsz, 1, emb_dim+enc_hid_dim
        rnn_input = torch.cat((embed, weighted), dim = -1)

        # output: (bsz, 1, hidden_dim)
        print(rnn_input.shape, embed.shape, weighted.shape, prev_hidden.shape)
        output, hidden = self.rnn(rnn_input, prev_hidden.unsqueeze(0))


        # print(output.shape, hidden.shape, embed.shape, )
        # assert (output == hidden).all()

        embed = embed.squeeze(1)
        output = output.squeeze(1)
        weighted = weighted.squeeze(1)
        
        # bsz, emb_dim + enc_hid_dim + dec_hid_dim
        output = torch.cat((output, weighted, embed), dim = 1)
        
        
        # [batch size, len_pre+len_post, dec_hid_dim]
        out_point = output.unsqueeze(1).repeat(1, enc_out.size(1), 1)

        # [batch size, len_pre+len_post, dec_hid_dim+enc_hid_dim]
        out_point = F.relu(torch.cat([enc_out, out_point], dim=-1))
        print(out_point.shape)
        out_point = self.fc_point(out_point)
        out_type = self.fc_type(output)

        return out_type, out_point, hidden.squeeze(0)
      


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
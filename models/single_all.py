import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy




class SingleAll(nn.Module):
    def __init__(self, args, vocab_size, embedding_dim, hidden_dim, mode):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = Encoder(self.embedding, embedding_dim, hidden_dim, hidden_dim)
        
        # self.decoder = Decoder(vocab_size, hidden_dim)
        self.fc_type = nn.Linear(hidden_dim, vocab_size)
        self.fc_point = nn.Linear(2*hidden_dim, 1)


    def forward(self, pre_seq, post_seq, *args, **kwargs):
        # pre_seq, post_seq = pre_seq.transpose(1, 0), post_seq.transpose(1, 0)
        len_pre = pre_seq.size(1)
        
        enc_out, enc_hidden = self.encoder(pre_seq, post_seq)
        # print(enc_out.shape, enc_hidden.shape)

        enc_point = enc_hidden.unsqueeze(1).repeat(1, enc_out.size(1), 1)
        out_point = F.relu(torch.cat([enc_out, enc_point], dim=-1))

        out_point = self.fc_point(out_point)
        out_type = self.fc_type(enc_hidden)

        return out_type, torch.sigmoid(out_point[:, :len_pre, :]), torch.sigmoid(out_point[:, len_pre:, :])


class SingleAllAttn(nn.Module):
    def __init__(self, args, vocab_size, embedding_dim, hidden_dim, mode):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = Encoder(self.embedding, embedding_dim, hidden_dim, hidden_dim)
        
        self.attn = Attention(hidden_dim, hidden_dim)
        self.decoder = Decoder(vocab_size, hidden_dim, hidden_dim, self.attn)
        
    def forward(self, pre_seq, post_seq, *args, **kwargs):

        len_pre = pre_seq.size(1)
        
        # enc_out: (bsz, len_pre+len_post, enc_hid_dim)
        # enc_hidden: (bsz, dec_hid_dim)
        enc_out, enc_hidden = self.encoder(pre_seq, post_seq)
        

        # enc_hidden: (bsz, vocab_size)
        out_type, out_point = self.decoder(enc_hidden, enc_out)
        
        return ( out_type, 
                torch.sigmoid(out_point[:, :len_pre, :]), 
                torch.sigmoid(out_point[:, len_pre:, :]) )




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
    def __init__(self, n_classes, enc_hid_dim, dec_hid_dim, attention, drop_p=0.5):
        super().__init__()
        
        self.dec_layer = nn.GRU(input_size=dec_hid_dim, hidden_size=dec_hid_dim, num_layers=1, dropout=drop_p, batch_first=True)
        
        self.fc1 = nn.Linear(dec_hid_dim, dec_hid_dim)
        self.fc2 = nn.Linear(dec_hid_dim+enc_hid_dim, dec_hid_dim)
        self.fc_type = nn.Linear(dec_hid_dim+enc_hid_dim, n_classes)
        self.fc_point = nn.Linear(dec_hid_dim+enc_hid_dim, 1)

        self.attention = attention

    def forward(self, hidden, encoder_outputs):

        # [batch size, src len]
        a = self.attention(hidden, encoder_outputs)
                
        # [batch size, 1, src len]
        a = a.unsqueeze(1)
        
        
        # [batch size, 1, enc hid dim]
        weighted = torch.bmm(a, encoder_outputs).squeeze(1)
        

        # [batch size, enc hid dim]
        hidden = torch.tanh(self.fc1(hidden))
        
        # print(output.shape, weighted.shape)

        #prediction = [batch size, output dim]
        hidden = torch.tanh(self.fc2(torch.cat((hidden, weighted), dim = 1)))


        # [batch size, len_pre+len_post, dec_hid_dim]
        out_point = hidden.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)

        # [batch size, len_pre+len_post, dec_hid_dim+enc_hid_dim]
        out_point = F.relu(torch.cat([encoder_outputs, out_point], dim=-1))
        
        # out_type = self.fc_type(torch.cat((hidden, weighted), dim = 1))
        out_point = self.fc_point(out_point)
        out_type = self.fc_type(hidden)
        
        
        return out_type, out_point



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


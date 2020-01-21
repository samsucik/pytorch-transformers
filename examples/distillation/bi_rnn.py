import torch
import torch.nn as nn
import torch.nn.functional as F

from distillation.base import init_embedding, fetch_embedding


class BiRNNModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        fc_size = config.fc_size
        init_embedding(self, config)
        self.n_layers = (config.n_layers_lstm if hasattr(config, "n_layers_lstm") else 1)
        self.bi_rnn = nn.LSTM(self.embedding_dim, self.hidden_size, self.n_layers, batch_first=True, bidirectional=True)        
        self.fc1 = nn.Linear(2 * self.hidden_size, fc_size)
        self.fc2 = nn.Linear(fc_size, config.n_classes)
        self.dropout = nn.Dropout(config.dropout)
        self.mode = config.mode

    def non_embedding_params(self):
        params = []
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                continue
            params.extend(p for p in m.parameters() if p.dim() == 2)
        return params

    def forward(self, sent_tuple):
        sent, sent_length = sent_tuple
        sent = fetch_embedding(self, self.mode, sent, squash=True)#.squeeze(1)
        
        sorted_len, sorted_ind = torch.sort(sent_length, descending=True)
        _, sorted_back_ind = torch.sort(sorted_ind)
        sorted_sent = sent[sorted_ind]
        # print("SS", sorted_sent.shape)
        try:
            packed = torch.nn.utils.rnn.pack_padded_sequence(sorted_sent, sorted_len, batch_first=True)
        except RuntimeError as e:
            sorted_len = torch.eq(sorted_len, 0).long() + sorted_len
            packed = torch.nn.utils.rnn.pack_padded_sequence(sorted_sent, sorted_len, batch_first=True)
        # print("packed", packed.data.shape)
        rnn_seq, rnn_out = self.bi_rnn(packed)
        # print("rseq", rnn_seq.data.shape)
        # print("rnn_out", rnn_out[0].shape)
        # https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM
        rnn_out = rnn_out[0].permute(1, 0, 2)[sorted_back_ind] # (B, DxL, H)
        # print("sorted back", rnn_out.shape)
        B = rnn_out.size()[0]
        DxL = rnn_out.size()[1]
        H = rnn_out.size()[2]
        L = self.n_layers if hasattr(self, "n_layers") else 1
        # print(rnn_out.size()[0], rnn_out.size()[1], rnn_out.size()[2])
        # rnn_out = rnn_out.contiguous().view(rnn_out.size()[0], rnn_out.size()[1] * rnn_out.size()[2])

        rnn_out = rnn_out.view(B, 2, L, H) # separate # of directions (2) and # of layers
        # print(rnn_out.shape)
        rnn_out = rnn_out[:, :, -1, :] # throw away all but last layer
        # print(rnn_out.shape)
        rnn_out = rnn_out.contiguous().view(B, 2*H) # reshape into 2D
        # print(rnn_out.shape)

        # rnn_out = rnn_out.permute(1, 0, 2).contiguous().view(self.n_layers, 2, B, H)[-1, :] \
        #                  .view(2, B, H).permute(1, 0, 2).view(B, 2*H)    
        # h_n.view(num_layers, num_directions, batch, hidden_size)

        # print("viewed", rnn_out.shape)
        # print("fc1", self.fc1)
        x = F.relu(self.fc1(rnn_out))
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits

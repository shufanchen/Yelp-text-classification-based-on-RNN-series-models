#coding=utf-8
import torch
#from torchtext.vocab import GloVe



class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,dropout, model_type='RNN'):
        super(Model, self).__init__()
        #    self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if model_type == 'LSTM':
            self.rnn = torch.nn.LSTM(input_size=self.input_size,
                                     hidden_size=self.hidden_size,
                                     num_layers=num_layers,
                                     dropout=dropout,
                                     batch_first=True)
        elif model_type == 'GRU':
            self.rnn = torch.nn.GRU(input_size=self.input_size,
                         hidden_size=self.hidden_size,
                         num_layers=num_layers,
                         dropout=dropout,
                         batch_first=True)
        else:
            self.rnn = torch.nn.RNN(input_size=self.input_size,
                         hidden_size=self.hidden_size,
                         num_layers=num_layers,
                         dropout=dropout,
                         batch_first=True)            
        self.linear1 = torch.nn.Linear(self.hidden_size, 5)
        #self.embedding = GloVe(name='6B', dim=self.input_size)
        self.embedding = torch.nn.Embedding(num_embeddings=46960, embedding_dim=input_size)
    def forward(self, input):#input格式必须是(batch_size,seqLen,input_Size
        input = input.squeeze(dim=-1)
        input = self.embedding(input)
        out, h_state = self.rnn(input)

        #out是这一批数据每个时间步的输出，shape(batch_size,seqLen,hidden_size)

        out = self.linear1(out[:, -1, :])  # 只保留最后一个时间步的输出，即每seqLen个样本只产生一个输出

        return out
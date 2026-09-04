import torch.nn as nn
import torch as t

# encoding and decoding은 동일하게 
class VanillaLSTM(nn.Module):
    def __init__(self, args):
        super(VanillaLSTM, self).__init__()
        
        self.f_length = args['f_length']
        self.traj_linear_hidden = args['traj_linear_hidden']
        self.lstm_encoder_size = args['lstm_encoder_size']
        self.out_length = args['out_length'] 

        self.dropout = nn.Dropout(args['dropout'])

        self.use_elu = args['use_elu']
        if self.use_elu:
            self.activation = nn.ELU()
        else:
            self.activation = nn.LeakyReLU(args['relu'])

        # Encoder: (Seq, Batch, 2) 입력 -> Hidden State 추출
        self.encoder = nn.Sequential(
            nn.Linear(self.f_length, self.traj_linear_hidden),
            self.activation,
            self.dropout,
            nn.LSTM(self.traj_linear_hidden, self.lstm_encoder_size)
        )
        
        # Decoder: Hidden State를 입력받아 미래 경로 (fut_len, 2) 생성
        self.decoder = Decoder(args)

    def forward(self, hist, va, cls):
        # hist shape: (maxlen, Batch, 2)
        # _, (hidden, _) = self.encoder(hist)

        hist = t.cat((hist, cls, va), -1)
        _, (h_n, c_n) = self.encoder(hist) # (31, 128, 64)

        # 마지막 레이어의 hidden state (Batch, Hidden_dim)
        # last_hidden = hidden[-1]
                
        # 미래 경로 생성 및 Reshape -> (fut_len, Batch, 2)
        fut_pred = self.decoder(h_n, c_n)
        return fut_pred
    

class Decoder(nn.Module):
    def __init__(self,args):
        super(Decoder,self).__init__()
        
        self.lstm_encoder_size = args['lstm_encoder_size']
        self.out_length = args['out_length']
        
        self.lstm = nn.LSTM(self.lstm_encoder_size, self.lstm_encoder_size)
        self.fc = nn.Linear(self.lstm_encoder_size, 2)
        
    def forward(self, h_n, c_n): # h_n : (1.batch,hidden)
        decoder_input = h_n.repeat(self.out_length, 1, 1)
        h, (_,_) = self.lstm(decoder_input, (h_n,c_n))
        output = self.fc(h)
        
        return output
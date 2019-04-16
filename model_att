import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import Arg

class SiaNetwork(nn.Module):

    def __init__(self, args,word2ix,word_matrix):
        super(SiaNetwork, self).__init__()

        self.args = args
        self.emb_num = word_matrix.size()[0]
        self.emb_dim = word_matrix.size()[1]
        
        self.word_emb =nn.Embedding(self.emb_num,self.emb_dim,padding_idx = word2ix['padding'])
        # initialize word embedding with GloVe
        self.word_emb.weight = nn.Parameter(torch.FloatTensor(word_matrix))
        # fine-tune the word embedding
        self.word_emb.weight.requires_grad = args.need_word_grad
        # <unk> vectors is randomly initialized
        nn.init.normal_(self.word_emb.weight.data[word2ix['unknow']])

        # BiLSTM encoder with shortcut connections
        self.Gru = nn.GRU(
                input_size=self.emb_dim,
                hidden_size=args.hidden_dim,
                num_layers=args.layers,
                dropout=args.dropout,  
                bidirectional=True,
                batch_first=True
            )

        # vector-based multi-head attention


        # fully-connected layers for classification
        self.fc1 = nn.Linear( 2*4*2 * args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear( 2*4*2 * args.hidden_dim + args.hidden_dim, args.hidden_dim)
        self.fc_out = nn.Linear(args.hidden_dim, args.class_size)
        self.relu = nn.ReLU()



    def forward(self, sentence1,sentence2,sen_lengths1,sen_lengths2):

        batch_sentence_emb1 = self.word_emb(sentence1)
        batch_sentence_emb2 = self.word_emb(sentence2)

        sorted_seq_lengths1, indices1 = torch.sort(sen_lengths1, descending=True)
        _, desorted_indices1 = torch.sort(indices1, descending=False)

        batch_sentence_emb1 =  batch_sentence_emb1[indices1]
        
        sorted_seq_lengths2, indices2 = torch.sort(sen_lengths2, descending=True)
        _, desorted_indices2 = torch.sort(indices2, descending=False)
        batch_sentence_emb2 = batch_sentence_emb2[indices2]
    
        
        Gru_in1 = pack(batch_sentence_emb1, sorted_seq_lengths1.cpu().numpy(), batch_first=True)
        Gru_in2 = pack(batch_sentence_emb2, sorted_seq_lengths2.cpu().numpy(), batch_first=True)

        # BiLSTM sequence encoder
        Gru_out1 ,hidden =self.Gru(Gru_in1)
        Gru_out2 ,hidden =self.Gru(Gru_in2)
        Gru_out1,_ = unpack(Gru_out1,batch_first=True)
        Gru_out2,_ = unpack(Gru_out2,batch_first=True)
        Gru_out1 = Gru_out1[desorted_indices1]
        Gru_out2 = Gru_out2[desorted_indices2]
 
        s1_vec  = pool(Gru_out1)
        s2_vec  = pool(Gru_out2)
        
        cat_feature = torch.cat([s1_vec, s2_vec, (s1_vec - s2_vec).abs(), s1_vec * s2_vec], dim=-1)
        # fully-connected layers
        out = self.fc1(cat_feature)
        out = self.relu(out)
        out = self.fc2(torch.cat([cat_feature, out], dim=-1))
        out = self.relu(out)
        out = self.fc_out(out)
        return out


def pool(x):
    max_pool = nn.AdaptiveMaxPool1d(1)
    avg_pool = nn.AdaptiveAvgPool1d(1)
    mx = max_pool(x.permute(0,2,1)).permute(0,2,1).squeeze(1)
    ax = avg_pool(x.permute(0,2,1)).permute(0,2,1).squeeze(1)
    return torch.cat([mx, ax], dim=-1)


import random
import numpy as np
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch_geometric as pyg
import torch.nn.functional as F
import pickle
from torch.nn import BatchNorm2d,BatchNorm1d,Dropout
from torch.utils.data import Dataset, DataLoader
import os 
os.environ['CUDA_LAUNCH_BLOCKING']='1' 
from tqdm import tqdm
import gc
from eviltransform import distance
import pickle
#import Geohash

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('model device:',device)
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor      #数据类型

"""dataset"""

with open('轨迹预测/dict_stay_xy.pkl','rb+') as f:     
    dict_stay_xy=pickle.load(f)

len_history=15
len_predict=15

with open('轨迹预测/train_data.pkl', 'rb') as f:
    train_data=pickle.load( f)
with open('轨迹预测/test_data.pkl', 'rb') as f:
    test_data=pickle.load( f)

"""transformer"""

class AddAndNorm(nn.Module):
    def __init__(self, d_model):
        super(AddAndNorm, self).__init__()
        
        self.layer_norm=nn.LayerNorm(d_model) # 减均值除标准差

    def forward(self, x, residual):
        return self.layer_norm(x+residual)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_head):
        super(ScaledDotProductAttention, self).__init__()
        self.d_head = d_head
        self.attention_dropout = nn.Dropout(p=0.1)
    def forward(self, q, k, v, mask=None):
        attention_weights = torch.matmul(q, k.transpose(-2, -1))  # 将倒数2和1维转置。相乘后维度(batch_size, n_heads, seq_len, seq_len)
        scaled_attention_weights = attention_weights / math.sqrt(self.d_head)  # (batch_size, n_heads, seq_len, seq_len)
        if mask is not None:
            scaled_attention_weights = scaled_attention_weights.masked_fill(mask == 0, float('-inf')) # (batch_size, n_heads, seq_len, seq_len)
        # Apply softmax over the last dimension which corresponds to attention weights for a key 
        scaled_attention_weights = nn.functional.softmax(scaled_attention_weights, dim=-1) # (batch_size, n_heads, seq_len, seq_len)
        scaled_attention_weights = self.attention_dropout(scaled_attention_weights) # (batch_size, n_heads, seq_len, seq_len)
        weighted_v = torch.matmul(scaled_attention_weights, v) # (batch_size, n_heads, seq_len, d_head)
        return weighted_v

class PositionWiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForwardNet, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        # Optional Dropout (not mentioned in the paper)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads= n_heads
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.dot_product_attention_layer= ScaledDotProductAttention(self.d_head)  
        self.W_0 = nn.Linear(d_model, d_model)
    def _split_into_heads(self, q,k,v):
        q= q.view(q.size(0), q.size(1), self.n_heads, self.d_head) # (batch_size, seq_len, n_heads, d_head)就是拆分d_model
        k= k.view(k.size(0), k.size(1), self.n_heads, self.d_head) # (batch_size, seq_len, n_heads, d_head)
        v= v.view(v.size(0), v.size(1), self.n_heads, self.d_head) # (batch_size, seq_len, n_heads, d_head)
        q= q.transpose(1,2) # (batch_size, n_heads, seq_len, d_head)
        k= k.transpose(1,2) # (batch_size, n_heads, seq_len, d_head)
        v= v.transpose(1,2) # (batch_size, n_heads, seq_len, d_head)
        return q,k,v
    def _concatenate_heads(self,attention_output):
        attention_output = attention_output.transpose(1,2).contiguous() # (batch_size, seq_len, n_heads, d_head)
        attention_output = attention_output.view(attention_output.size(0), attention_output.size(1), -1) # (batch_size, seq_len, n_heads * d_head)
        return attention_output
    def forward(self, q, k, v, mask=None):
        q,k,v= self._split_into_heads(q,k,v) # (batch_size, n_heads, seq_len, d_head)
        attention_output = self.dot_product_attention_layer(q, k, v, mask) # (batch_size, n_heads, seq_len, d_head)
        attention_output = self._concatenate_heads(attention_output) # (batch_size, seq_len, n_heads * d_head)
        attention_output = self.W_0(attention_output) # (batch_size, seq_len, d_model)
        return attention_output 

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_proba):
        super(TransformerEncoderBlock, self).__init__()
        self.W_q = nn.Linear(d_model, d_model) 
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)        
        # 多头注意力
        self.mha_layer=MultiHeadAttention(d_model, n_heads)
        self.dropout_layer_1=nn.Dropout(dropout_proba)
        self.add_and_norm_layer_1 = AddAndNorm(d_model)        
        #前向
        self.ffn_layer = PositionWiseFeedForwardNet(d_model, d_ff)
        self.dropout_layer_2=nn.Dropout(dropout_proba)
        self.add_and_norm_layer_2 = AddAndNorm(d_model)
    def forward(self, x, mask=None):
        q = self.W_q(x) # (batch_size, src_seq_len, d_model)
        k = self.W_k(x) # (batch_size, src_seq_len, d_model)
        v = self.W_v(x) # (batch_size, src_seq_len, d_model)
        mha_out = self.mha_layer(q, k, v, mask) # (batch_size, src_seq_len, d_model)
        mha_out= self.dropout_layer_1(mha_out) # (batch_size, src_seq_len, d_model)
        mha_out = self.add_and_norm_layer_1(x, mha_out) # (batch_size, src_seq_len, d_model)
        ffn_out = self.ffn_layer(mha_out) # (batch_size, src_seq_len, d_model)
        ffn_out= self.dropout_layer_2(ffn_out) # (batch_size, src_seq_len, d_model)
        ffn_out = self.add_and_norm_layer_2(mha_out, ffn_out)  # (batch_size, src_seq_len, d_model)
        return ffn_out

class TransformerEncoder(nn.Module):
    def __init__(self, n_blocks, n_heads, d_model, d_ff, dropout_proba=0.1):
        super(TransformerEncoder, self).__init__()
        self.encoder_blocks=nn.ModuleList([TransformerEncoderBlock(d_model, n_heads, d_ff, dropout_proba) for _ in range(n_blocks)])
    def forward(self, x, mask=None):
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, mask)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=len_history, dropout_proba=0.1):
        super(PositionalEncoding, self).__init__()
        self.max_seq_len=max_seq_len
        self.d_model=d_model
        pe_table=self.get_pe_table()
        self.register_buffer('pe_table' , pe_table)
        self.dropout=nn.Dropout(dropout_proba) 
    def get_pe_table(self):
        position_idxs=torch.arange(self.max_seq_len).unsqueeze(1) 
        embedding_idxs=torch.arange(self.d_model).unsqueeze(0)        
        angle_rads = position_idxs * 1/torch.pow(10000, torch.true_divide((2*(embedding_idxs//2)),self.d_model))
        angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])
        pe_table = angle_rads.unsqueeze(0) # So we can apply it to a batch
        return pe_table
    def forward(self, embeddings_batch):
        seq_len = embeddings_batch.size(1)
        pe_batch = self.pe_table[:, :seq_len].clone().detach().repeat(embeddings_batch.size(0), 1, 1)   
        return self.dropout(embeddings_batch + pe_batch)

class TransformerEncoderExtractor(nn.Module):
    def __init__(self,d_model, n_blocks, n_heads, d_ff, dropout_proba, trg_vocab_size=1,
                geohash_dim=45, hidden_geohash_dim=16,if_pos_emb=True,shuzhi_dim=5,leibie_dim=4,nd_dim=128,hidden_nd_dim=16):
        super(TransformerEncoderExtractor, self).__init__()
        self.dropout_proba = dropout_proba
        self.d_model=d_model
        self.if_pos_emb=if_pos_emb
        self.src_pos_embedding= PositionalEncoding(d_model)   # 位置编码。
        self.encoder= TransformerEncoder(n_blocks, n_heads, d_model, d_ff, dropout_proba)
        self.linear = nn.Linear(shuzhi_dim+leibie_dim+hidden_geohash_dim+hidden_nd_dim,d_model)
        self.init_with_xavier()
        self.geohash_linear = nn.Linear(geohash_dim, hidden_geohash_dim)
        self.nd_linear = nn.Linear(nd_dim, hidden_nd_dim)
    def encode(self,shuzhi,leibie,geohash,nd,src_mask=None):
        geohash=self.geohash_linear(geohash)
        nd=self.nd_linear(nd)
        src_feature=torch.cat((shuzhi,leibie,geohash,nd), dim=-1)
        src_feature=self.linear(src_feature)
        if self.if_pos_emb:
            src_feature = self.src_pos_embedding(src_feature)
        encoder_outputs = self.encoder(src_feature, src_mask) # (batch_size, src_seq_len, d_model)  
        return encoder_outputs
    def forward(self, shuzhi,leibie,geohash,nd,src_mask=None):
        encoder_outputs= self.encode(shuzhi,leibie,geohash,nd,src_mask) # (batch_size, src_seq_len, d_model)
        return encoder_outputs     
    def init_with_xavier(self): 
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class MachineTranslationTransformer(nn.Module):
    def __init__(self, d_model,n_blocks,n_heads,d_ff, dropout_proba,trg_vocab_size=1,
                geohash_dim=45, hidden_geohash_dim=16,if_pos_emb=True,shuzhi_dim=5,leibie_dim=4,nd_dim=128,hidden_nd_dim=16):
        super(MachineTranslationTransformer, self).__init__()
        self.transformer_encoder_decoder=TransformerEncoderExtractor(
            d_model,
            n_blocks,
            n_heads,
            d_ff,
            dropout_proba,
            trg_vocab_size,
            geohash_dim,
            hidden_geohash_dim,
            if_pos_emb,
            shuzhi_dim,
            leibie_dim,
            nd_dim,
            hidden_nd_dim
        )
    def _get_pad_mask(self, token_ids, pad_idx=-99999):
        pad_mask= (token_ids != pad_idx).unsqueeze(-2) # (batch_size, 1, seq_len)
        return pad_mask.unsqueeze(1)   
    def forward(self, shuzhi,leibie,geohash,nd, src_mask=None):
        return self.transformer_encoder_decoder(shuzhi,leibie,geohash,nd, src_mask)

class MLP_final(nn.Module):
    def __init__(self, d_model1,d_model2):
        super(MLP_final, self).__init__()
        self.linear=nn.Linear(d_model1+d_model2,1)
        self.activate=nn.ReLU()
    def forward(self,x):
        x=self.linear(x)
        x=self.activate(x)
        return x

class MLP_final_add(nn.Module):
    def __init__(self, d_model):
        super(MLP_final_add, self).__init__()
        self.linear=nn.Linear(d_model,1)
        self.activate=nn.ReLU()
    def forward(self,x):
        x=self.linear(x)
        x=self.activate(x)
        return x

class MLP_final_aux(nn.Module):
    def __init__(self, d_model,d_model_above):
        super(MLP_final_aux, self).__init__()
        self.linear=nn.Linear(d_model,d_model_above)
    def forward(self,x):
        x=self.linear(x)
        return x

"""LSTM-Attention-preferences"""    
    
from torch.nn import functional as F
class DRRnnDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, use_tanh=False, tanh_exploration=10, decode_type="greedy", device="cuda"):
        super(DRRnnDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_tanh = use_tanh
        self.tanh_exploration = tanh_exploration
        # For online-deployment, use RNN instead of LSTM for faster computation.
        self.rnn = nn.RNNCell(input_size, hidden_size)        
        self.pointer = PointerNetAttention(hidden_size=hidden_size, use_tanh=use_tanh,
                                           tanh_exploration=tanh_exploration)        
        self.decode_type = decode_type
        self.device = device
        self.flag=True
        self.flag2=True        
        self.linear_dim=16
        # Load matrices MD and MC
        self.M_juli=torch.Tensor(np.load('轨迹预测/距离.npy')).to(device)
        self.M_zhuanyi=torch.Tensor(np.load('轨迹预测/转移.npy')).to(device)
        self.linear_mat_juli = nn.Linear(len_predict, len_predict)
        self.linear_mat_zhuanyi = nn.Linear(len_predict, len_predict)
        self.final_linear=nn.Linear(len_predict, len_predict)
    def forward(self, decoder_initial, encoder_h, encoder_h_mean, encoder_h_transpose,block_id):
        batch_size, seq_len, hidden_size = encoder_h.size()
        assert hidden_size == self.hidden_size
        current_ptr_masks=torch.zeros((batch_size,seq_len)).bool().to(self.device)     
        ptr_logits = []
        ptr_probs = []
        ptr_selections = []
        rnn_hidden_state=[]
        if self.flag:
            print('org current_ptr_masks',current_ptr_masks)
        for i in (range(seq_len)):
            logits, probs, encoder_h_mean = self.recurrence(decoder_initial, encoder_h_mean, 
                                                            encoder_h_transpose,block_id, current_ptr_masks)
            max_idx = self.select_ptr(probs, current_ptr_masks.bool(), seq_len)
            ptr_logits.append(logits)#[128,13]
            ptr_probs.append(probs)#[128,13]
            ptr_selections.append(max_idx)##[128]            
            max_idx = max_idx.detach().unsqueeze(1)#[128]->[128,1]
            tmp_mask=(torch.arange(seq_len).unsqueeze(0).to(device) == max_idx)
            current_ptr_masks = current_ptr_masks | tmp_mask         
            gather_idx = max_idx.contiguous().view(batch_size, 1, 1).expand(batch_size, 1, self.input_size)  # [1, batch_size, input_size]  #[128,1,256]
            decoder_initial = torch.gather(encoder_h, 1, gather_idx).squeeze(1)
            rnn_hidden_state.append(encoder_h_mean)
        if self.flag:
            self.flag=False
        # (batch_size,seq_len,seq_len),(batch_size,seq_len,seq_len),(batch_size,seq_len),(batch_size,seq_len,hidden_dim)
        return torch.stack(ptr_logits, 1), torch.stack(ptr_probs, 1), torch.stack(ptr_selections, 1).squeeze(-1),torch.stack(rnn_hidden_state,1)
    def recurrence(self, decoder_initial, encoder_h_mean, encoder_h_transpose,block_id, current_ptr_masks=None):
        hy = self.rnn(decoder_initial, encoder_h_mean)
        e, logits = self.pointer(hy, encoder_h_transpose)
        logits=F.relu(logits)
        expanded_x = block_id.unsqueeze(2).long()  
        expanded_x_transposed = expanded_x.transpose(1, 2).long()  
        mat_juli = self.M_juli[expanded_x, expanded_x_transposed]
        mat_zhuanyi = self.M_zhuanyi[expanded_x, expanded_x_transposed]
        mat_juli = torch.sigmoid(self.linear_mat_juli(mat_juli))
        mat_juli = F.normalize(mat_juli,dim=-1)        
        mat_zhuanyi = torch.sigmoid(self.linear_mat_zhuanyi(mat_zhuanyi))
        mat_zhuanyi = F.normalize(mat_zhuanyi,dim=-1)   
        final_logits = logits+torch.bmm(logits.unsqueeze(1), mat_juli).squeeze(1)+\
                              torch.bmm(logits.unsqueeze(1), mat_zhuanyi).squeeze(1)
        if current_ptr_masks is not None:
            final_logits[current_ptr_masks] = -1e9 
        probs = F.softmax(final_logits,dim=-1)
        if self.flag2:
            self.flag2=False        
        return final_logits, probs, hy
    def select_ptr(self, probs, ptr_masks, seq_len):
        _, max_idx = probs.max(dim=1)  
        max_idx[ptr_masks.all(dim=1)] = seq_len - 1
        return max_idx


class DR(nn.Module):
    def __init__(self, hidden_size, device="cuda"):
        super(DR, self).__init__()
        self.hidden_size=hidden_size
        self.device = device
        self.decoder = DRRnnDecoder(self.hidden_size, self.hidden_size, device=self.device)
        std = 1. / math.sqrt(self.hidden_size)
        self.decoder_initial = nn.Parameter(torch.randn(self.hidden_size)).to(self.device) 
        self.linear1=nn.Linear(self.hidden_size*2,self.hidden_size)
        self.linear2=nn.Linear(self.hidden_size,1)
        self.activate=nn.ReLU()        
    def forward(self, encoder_h,block_id):
        encoder_h_mean=encoder_h.mean(dim=1)        
        batch_size= encoder_h.size()[0]        
        decoder_initial = self.decoder_initial.unsqueeze(0).repeat(batch_size, 1)    
        ptr_logits, ptr_probs, ptr_selections,rnn_hidden_state = self.decoder(decoder_initial, encoder_h,
                                                 encoder_h_mean, encoder_h.transpose(0, 1),block_id)
        batch_size,seq_len,hidden_dim=encoder_h.size()
        indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        encoder_h_chongzu = torch.gather(encoder_h, 1, ptr_selections.unsqueeze(-1).expand(-1, -1, hidden_dim))        
        encoder_h_final=torch.cat((encoder_h_chongzu,rnn_hidden_state),dim=-1)
        final_pred1=self.linear1(encoder_h_final)
        final_pred2=self.linear2(final_pred1)
        activate_final_pred=self.activate(final_pred2)    
        return ptr_probs, activate_final_pred

class PointerNetAttention(nn.Module):
    def __init__(self, hidden_size, query_input_size=None, ref_input_size=None, use_tanh=False, tanh_exploration=10, device="cpu"):
        super(PointerNetAttention, self).__init__()
        if query_input_size is not None:
            self.query_input_size = query_input_size
        else:
            self.query_input_size = hidden_size
        if ref_input_size is not None:
            self.ref_input_size = ref_input_size
        else:
            self.ref_input_size = hidden_size
        self.hidden_size = hidden_size
        self.use_tanh = use_tanh
        self.tanh_exploration = tanh_exploration
        self.project_query = nn.Linear(self.query_input_size, hidden_size)
        self.project_ref = nn.Conv1d(self.ref_input_size, hidden_size, 1, 1)        
        self.tanh = nn.Tanh()
        self.v = nn.Parameter(torch.FloatTensor(hidden_size))
        self.v.data.uniform_(-(1. / math.sqrt(hidden_size)), 1. / math.sqrt(hidden_size))
    def forward(self, query, ref):
        ref = ref.permute(1, 2, 0)  
        q = self.project_query(query).unsqueeze(2)  
        e = self.project_ref(ref)   
        expanded_q = q.repeat(1, 1, e.size(2))  
        expanded_v = self.v.unsqueeze(0)\
            .expand(expanded_q.size(0), self.hidden_size).unsqueeze(1)   
        u = torch.bmm(expanded_v, self.tanh(expanded_q + e)).squeeze(1)     
        if self.use_tanh:
            return e, self.tanh_exploration * self.tanh(u)
        else:
            return e, u

"""pattern-memory"""
        
import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryNetwork(nn.Module):
    def __init__(self, input_size, output_size, mem_size=8, mem_dim=64,):
        super(MemoryNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.mem_size = mem_size
        self.mem_dim = mem_dim
        self.hid_size = mem_dim       
        self.query_net = nn.Linear(input_size, self.hid_size)
        self.memory = nn.Parameter(torch.randn(mem_size, mem_dim))
        self.out_net = nn.Linear(input_size+mem_dim, output_size)
        self.flag=False
    def forward(self, x):
        query = self.query_net(x)   
        sim=torch.matmul(query, self.memory.t())    
        sim = F.softmax(sim, dim=-1)       
        out = torch.matmul(sim, self.memory)    
        out=torch.cat([x,out],dim=-1)    
        if not self.flag:    
            self.flag=True       
        out = self.out_net(out)   
        return out
    
def cal_dis(A,B):
    latA,lngA=A[0],A[1]
    latB,lngB=B[0],B[1]
    return distance(latA, lngA, latB, lngB)    

"""setting"""

history_model = MachineTranslationTransformer(
    d_model=64,     
    n_blocks=2,    
    n_heads=4,
    d_ff=64, 
    dropout_proba=0.1,
    trg_vocab_size=1,         
    geohash_dim=45, 
    hidden_geohash_dim=16,
    if_pos_emb=True,      
    shuzhi_dim=5,
    leibie_dim=4+1,
    nd_dim=128,
    hidden_nd_dim=16
).to(device)
todel_model = MachineTranslationTransformer(
    d_model=64,     
    n_blocks=2,     
    n_heads=4,
    d_ff=64, 
    dropout_proba=0.1,
    trg_vocab_size=1,         
    geohash_dim=45, 
    hidden_geohash_dim=16,
    if_pos_emb=False,      
    shuzhi_dim=5+1+1,        
    leibie_dim=4+1,
    nd_dim=128,
    hidden_nd_dim=16
).to(device)
DR_model=DR(hidden_size=64*4).to(device)
memory_model=MemoryNetwork(
                        input_size=64*2,
                        output_size=64*4,
                        mem_size=20, 
                        mem_dim=64*2,
                        ).to(device)    
    
trainloader = DataLoader(
    train_data,
    batch_size=256,
    shuffle=True,
    num_workers=8
    )
testloader = DataLoader(
    test_data,
    batch_size=256,
    shuffle=False,    #设置为false
    num_workers=8
    )

learning_rate=0.001    
optimizer = torch.optim.Adam(list(history_model.parameters())+list(todel_model.parameters())+\
                             list(DR_model.parameters())+list(memory_model.parameters())
                             , learning_rate, 
                            )

criteria = nn.MSELoss()             
criteria_cel = nn.CrossEntropyLoss()
def percentage_error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res

def mape(y_true, y_pred): 
    return np.mean(np.abs(percentage_error((y_true), (y_pred)))) * 100
    
seeds=[random.randint(0, 1000) for _ in range(5)]
for seed_id in seeds:
    torch.manual_seed(seed_id)
    torch.cuda.manual_seed_all(seed_id)
    np.random.seed(seed_id)
    random.seed(seed_id)
        
    """inference"""
    
    train_loss_list = []
    train_mape_list = []
    val_loss_list = []
    val_mape_list = []
    train_cel_list = []
    val_cel_list = []
    best_val_mape=100000
    for epochs in tqdm(range(100)):
        epoch_train_loss_list=[]
        epoch_val_loss_list=[]
        epoch_train_cel_list=[]
        epoch_val_cel_list=[]
        epoch_train_mape_list=[]
        epoch_val_mape_list=[]    
        history_model.train()
        todel_model.train()
        DR_model.train()
        memory_model.train()
        for i, data in (enumerate(trainloader)):
            optimizer.zero_grad()
            #历史
            ls_shuzhi,ls_leibie,ls_jingwei,ls_geohash,ls_nd=[],[],[],[],[]
            for k in range(len_history):
                shuzhi,leibie,jingwei,geohash,nd=data[k]        
                shuzhi=torch.stack(shuzhi,dim=1)
                leibie=torch.stack(leibie,dim=1)
                jingwei=torch.stack(jingwei,dim=1)
                geohash=torch.stack(geohash,dim=1)
                nd=torch.stack(nd,dim=1)
                ls_shuzhi.append(shuzhi)
                ls_leibie.append(leibie)
                ls_jingwei.append(jingwei)
                ls_geohash.append(geohash)
                ls_nd.append(nd)
            shuzhi1=torch.stack(ls_shuzhi,dim=1).float().to(device)
            leibie1=torch.stack(ls_leibie,dim=1).float().to(device)
            jingwei1=torch.stack(ls_jingwei,dim=1).float().to(device)
            geohash1=torch.stack(ls_geohash,dim=1).float().to(device)
            nd1=torch.stack(ls_nd,dim=1).float().to(device)
            his=history_model(shuzhi1,leibie1,geohash1,nd1)
            #未来
            ls_shuzhi,ls_leibie,ls_jingwei,ls_geohash,ls_nd=[],[],[],[],[]
            for k in range(len_predict):
                shuzhi,leibie,jingwei,geohash,nd=data[len_history+k]
                shuzhi=torch.stack(shuzhi,dim=1)
                leibie=torch.stack(leibie,dim=1)
                jingwei=torch.stack(jingwei,dim=1)
                geohash=torch.stack(geohash,dim=1)
                nd=torch.stack(nd,dim=1)
                ls_shuzhi.append(shuzhi)
                ls_leibie.append(leibie)
                ls_jingwei.append(jingwei)
                ls_geohash.append(geohash)
                ls_nd.append(nd)
            shuzhi2=torch.stack(ls_shuzhi,dim=1).float().to(device)
            leibie2=torch.stack(ls_leibie,dim=1).float().to(device)
            jingwei2=torch.stack(ls_jingwei,dim=1).float().to(device)
            geohash2=torch.stack(ls_geohash,dim=1).float().to(device)
            nd2=torch.stack(ls_nd,dim=1).float().to(device)
    
            songda_ac=shuzhi1[:,-1,-1].unsqueeze(1)
            songda_ai=shuzhi2[:,:,-1]
            songda_times=songda_ai-songda_ac  
            to_deliver_pattern=[]
            ground_truth=songda_times.reshape(-1)  
            A_expanded=-1*(shuzhi1[:,-1,-1].unsqueeze(1)-shuzhi2[:,:,0]).unsqueeze(2)   
            B_expanded = shuzhi1[:,-1,-1].unsqueeze(1).unsqueeze(2).repeat(1, len_predict, 1)   
            C_expanded=-1*(shuzhi1[:,-1,-1].unsqueeze(1)-shuzhi2[:,:,1]).unsqueeze(2)  
            shuzhi2 = torch.cat((shuzhi2[:,:,:-2],A_expanded,C_expanded, B_expanded), dim=-1)
            tmp_jingwei2=jingwei2.reshape(-1,jingwei2.size()[-1])
            tmp_jingwei1=jingwei1[:,-1,:].repeat_interleave(len_predict, dim=0).reshape(-1, jingwei1.size()[-1])
            tmp_geohash=geohash2.reshape(-1,geohash2.size()[-1])
            tmp_nd=nd2.reshape(-1,nd2.size()[-1])
            tmp_dis=list(map(cal_dis,tmp_jingwei1,tmp_jingwei2))
            tmp_dis=torch.Tensor(tmp_dis).unsqueeze(1).to(device)
            tmp_dis=tmp_dis.reshape(shuzhi2.size()[0],shuzhi2.size()[1],1)
            shuzhi2=torch.cat((shuzhi2,tmp_dis),dim=-1)
            todel=todel_model(shuzhi2,leibie2,geohash2,nd2)
            his_ac=his[:,-1,:].repeat_interleave(len_history, dim=0).reshape(his.size())
            fus=torch.cat((his_ac,todel),dim=-1)
            fus=memory_model(fus)
            block_id=leibie2[:,:,-2]
            ptr_probs, y_pred=DR_model(fus,block_id)
            y_pred=y_pred.squeeze().reshape(-1)
            mask=leibie2[:,:,-1].reshape(-1)
            batch_size,seq_len,_=ptr_probs.size()
            ptr_probs = ptr_probs.reshape(batch_size * seq_len, seq_len)
            indices = torch.arange(seq_len)
            indices = torch.unsqueeze(indices, dim=0).expand(batch_size, -1).reshape(-1).to(device)
            loss_cel=criteria_cel(ptr_probs,indices)
            loss = criteria(y_pred[mask==1],ground_truth[mask==1])
            mapes=mape(ground_truth[mask==1].detach().cpu().numpy(), y_pred[mask==1].detach().cpu().numpy())
            epoch_train_loss_list.append(loss.item())
            epoch_train_cel_list.append(loss_cel.item())
            epoch_train_mape_list.append(mapes)
            loss_final=loss+loss_cel
            loss_final.backward()
            optimizer.step()
        train_loss_list.append(np.mean(epoch_train_loss_list))
        train_cel_list.append(np.mean(epoch_train_cel_list))
        train_mape_list.append(np.mean(epoch_train_mape_list))
    
        cnt=0
        y_ps=[]
        y_ts=[]
        y_mask=[]
        y_indices=[]
        y_probs=[]
        with torch.no_grad():        
            history_model.eval()
            todel_model.eval()
            DR_model.eval()
            memory_model.eval()
            for i, data in (enumerate(testloader)):
                #历史
                ls_shuzhi,ls_leibie,ls_jingwei,ls_geohash,ls_nd=[],[],[],[],[]
                for k in range(len_history):
                    shuzhi,leibie,jingwei,geohash,nd=data[k]         
                    shuzhi=torch.stack(shuzhi,dim=1)
                    leibie=torch.stack(leibie,dim=1)
                    jingwei=torch.stack(jingwei,dim=1)
                    geohash=torch.stack(geohash,dim=1)
                    nd=torch.stack(nd,dim=1)
                    ls_shuzhi.append(shuzhi)
                    ls_leibie.append(leibie)
                    ls_jingwei.append(jingwei)
                    ls_geohash.append(geohash)
                    ls_nd.append(nd)
                shuzhi1=torch.stack(ls_shuzhi,dim=1).float().to(device)
                leibie1=torch.stack(ls_leibie,dim=1).float().to(device)
                jingwei1=torch.stack(ls_jingwei,dim=1).float().to(device)
                geohash1=torch.stack(ls_geohash,dim=1).float().to(device)
                nd1=torch.stack(ls_nd,dim=1).float().to(device)
                his=history_model(shuzhi1,leibie1,geohash1,nd1)
                #未来
                ls_shuzhi,ls_leibie,ls_jingwei,ls_geohash,ls_nd=[],[],[],[],[]
                for k in range(len_predict):
                    shuzhi,leibie,jingwei,geohash,nd=data[len_history+k]
                    shuzhi=torch.stack(shuzhi,dim=1)
                    leibie=torch.stack(leibie,dim=1)
                    jingwei=torch.stack(jingwei,dim=1)
                    geohash=torch.stack(geohash,dim=1)
                    nd=torch.stack(nd,dim=1)
                    ls_shuzhi.append(shuzhi)
                    ls_leibie.append(leibie)
                    ls_jingwei.append(jingwei)
                    ls_geohash.append(geohash)
                    ls_nd.append(nd)
                shuzhi2=torch.stack(ls_shuzhi,dim=1).float().to(device)
                leibie2=torch.stack(ls_leibie,dim=1).float().to(device)
                jingwei2=torch.stack(ls_jingwei,dim=1).float().to(device)
                geohash2=torch.stack(ls_geohash,dim=1).float().to(device)
                nd2=torch.stack(ls_nd,dim=1).float().to(device)
                songda_ac=shuzhi1[:,-1,-1].unsqueeze(1)
                songda_ai=shuzhi2[:,:,-1]
                songda_times=songda_ai-songda_ac  
                to_deliver_pattern=[]
                ground_truth=songda_times.reshape(-1,1)  
                A_expanded=-1*(shuzhi1[:,-1,-1].unsqueeze(1)-shuzhi2[:,:,0]).unsqueeze(2)   
                B_expanded = shuzhi1[:,-1,-1].unsqueeze(1).unsqueeze(2).repeat(1, len_predict, 1)   
                C_expanded=-1*(shuzhi1[:,-1,-1].unsqueeze(1)-shuzhi2[:,:,1]).unsqueeze(2)   
                shuzhi2 = torch.cat((shuzhi2[:,:,:-2],A_expanded,C_expanded, B_expanded), dim=-1)
                tmp_jingwei2=jingwei2.reshape(-1,jingwei2.size()[-1])
                tmp_jingwei1=jingwei1[:,-1,:].repeat_interleave(len_predict, dim=0).reshape(-1, jingwei1.size()[-1])
                tmp_geohash=geohash2.reshape(-1,geohash2.size()[-1])
                tmp_nd=nd2.reshape(-1,nd2.size()[-1])
                tmp_dis=list(map(cal_dis,tmp_jingwei1,tmp_jingwei2))
                tmp_dis=torch.Tensor(tmp_dis).unsqueeze(1).to(device)
                tmp_dis=tmp_dis.reshape(shuzhi2.size()[0],shuzhi2.size()[1],1)
                shuzhi2=torch.cat((shuzhi2,tmp_dis),dim=-1)
                todel=todel_model(shuzhi2,leibie2,geohash2,nd2)
                his_ac=his[:,-1,:].repeat_interleave(len_history, dim=0).reshape(his.size())
                fus=torch.cat((his_ac,todel),dim=-1)
                fus=memory_model(fus)
                block_id=leibie2[:,:,-2]
                ptr_probs, y_pred=DR_model(fus,block_id)           
                y_pred=y_pred.squeeze().reshape(-1)#
                batch_size,seq_len,_=ptr_probs.size()
                ptr_probs = ptr_probs.reshape(batch_size * seq_len, seq_len)
                indices = torch.arange(seq_len)
                indices = torch.unsqueeze(indices, dim=0).expand(batch_size, -1).reshape(-1).to(device)
                y_indices.append(indices)
                y_probs.append(ptr_probs)
                y_ps+=y_pred.squeeze().tolist()
                y_ts+=ground_truth.squeeze().tolist()            
                mask=leibie2[:,:,-1]
                y_mask+=mask.squeeze().reshape(-1).tolist()
            y_indices=torch.cat(y_indices,dim=0)
            y_probs=torch.cat(y_probs,dim=0)
            loss_cel=criteria_cel(ptr_probs,indices)
            y_ts_mask,y_ps_mask=[],[]
            if(len(y_ts)!=len(y_mask) or len(y_ps)!=len(y_ts)):
                raise ValueError("mask len wrong")
            for i in range(len(y_mask)):
                if y_mask[i]==1:
                    y_ts_mask.append(y_ts[i])
                    y_ps_mask.append(y_ps[i])
            actual_arr = np.array(y_ts_mask)
            predicted_arr = np.array(y_ps_mask)
            rmse = np.sqrt(np.mean((actual_arr - predicted_arr) ** 2))
            my_mape = mape(actual_arr ,predicted_arr)
            val_loss_list.append(rmse)
            val_mape_list.append(my_mape)

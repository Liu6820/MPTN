import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from model.tools import ccorr_new, cconv, ccorr,rotate, cconv_new
import torch.nn.functional as F
import math

class MPT(MessagePassing):
    def __init__(self, in_channels, out_channels, rel_dim, drop, bias, op, beta, num_head=1, final_layer=False):
        super(Transformer, self).__init__(aggr = "add", node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.op = op
        self.bias = bias
        self.head = num_head
        self.final_layer = final_layer
        self.beta = beta
        self.w_in = torch.nn.Linear(in_channels, out_channels)
        self.w_out = torch.nn.Linear(in_channels, out_channels)
        self.w_res = torch.nn.Linear(in_channels, out_channels)
                
        self.lin_key = torch.nn.Linear(in_channels, num_head*out_channels, bias=bias)
        self.lin_query = torch.nn.Linear(in_channels, num_head*out_channels, bias=bias)
        # self.lin_value = torch.nn.Linear(in_channels, num_head*out_channels, bias=bias)
        # self.loop_rel = torch.nn.Parameter(torch.Tensor(1, rel_dim)).cuda()
        # torch.nn.init.xavier_normal_(self.loop_rel)

        if final_layer:
            self.w_rel = torch.nn.Linear(rel_dim, out_channels).cuda()
        else:
            self.w_rel = torch.nn.Linear(rel_dim, num_head*out_channels).cuda()

        self.drop =drop
        self.dropout = torch.nn.Dropout(drop)
        
        self.bn = torch.nn.BatchNorm1d(out_channels).to(torch.device("cuda"))
        self.activation = torch.nn.Tanh()

        if bias:
            self.register_parameter("bias_value", torch.nn.Parameter(torch.zeros(out_channels)))
    
    def forward(self, x, edge_index, edge_type, rel_emb, pre_alpha=None):
        
        #print(x.shape, edge_index.shape, edge_type.shape, edge_type.shape)
        out = self.propagate(edge_index=edge_index, x=x, edge_type=edge_type, rel_emb=rel_emb, pre_alpha=pre_alpha)    
        
        loop_res = self.w_res(x).view(-1, self.head, self.out_channels)
        out = self.dropout(out) + self.dropout(loop_res)
        
        if self.final_layer:
            out = out.mean(dim=1)
        else:
            out = out.view(-1, self.head*self.out_channels)
        
        out = self.activation(out)    
        out = self.bn(out)
        #print(rel_emb.shape)
        #return out, self.w_rel(rel_emb), self.alpha.detach()
        return out, self.w_rel(rel_emb)

    def message(self, x_i, x_j, edge_type, rel_emb, ptr, index, size_i, pre_alpha):
        #print(x_i==x_j)
        #print(x_i.shape, x_j.shape, rel_emb.shape)
        
        rel_emb = torch.index_select(rel_emb, 0, edge_type)
        xj_rel = self.rel_transform(x_j, rel_emb)
       # print(xj_rel.shape)
        num_edge = xj_rel.size(0)//2

        in_message = xj_rel[:num_edge]
        out_message = xj_rel[num_edge:]
        
        trans_in = self.w_in(in_message)
        trans_out = self.w_out(out_message)
        out = torch.cat((trans_in, trans_out), dim=0).view(-1, self.head, self.out_channels)
        #print(out.shape)
        
        query = self.lin_query(x_i).view(-1, self.head, self.out_channels)
        key = self.lin_key(xj_rel).view(-1, self.head, self.out_channels)

        alpha = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.drop, training=self.training)

        if pre_alpha!=None and self.beta != 0:
            self.alpha = alpha*(1-self.beta) + pre_alpha*(self.beta)
        else:
            self.alpha = alpha
        # out = self.lin_value(xj_rel).view(-1, self.head, self.out_channels)
        out *= self.alpha.view(-1, self.head, 1)

        return out

    def update(self, aggr_out):
        return aggr_out

    def rel_transform(self, ent_embed, rel_emb):
        
        if self.op == 'corr':
            trans_embed  = ccorr(ent_embed, rel_emb)
        elif self.op == 'sub':
            trans_embed = ent_embed - rel_emb
        elif self.op == 'mult':
            trans_embed = ent_embed * rel_emb
        elif self.op == "corr_new":
            trans_embed = ccorr_new(ent_embed, rel_emb)
        elif self.op == "conv":
            trans_embed = cconv(ent_embed, rel_emb)
        elif self.op == "conv_new":
            trans_embed = cconv_new(ent_embed, rel_emb)
        else:
            raise NotImplementedError      
        return trans_embed

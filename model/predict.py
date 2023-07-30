from model.tools import *
from model.GPKG_conv import GPKGConv
from model.GPKG_conv_basis import GPKGConvBasis
from model.ComGATv3 import CompGATv3
from model.SupConLoss import SupConLoss
from model.Compgat import CompGAT
from model.Transformer import Transformer
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class BaseModel(torch.nn.Module):
    def __init__(self, params):
        super(BaseModel, self).__init__()

        self.p = params
        self.act = torch.tanh
        self.bceloss = torch.nn.BCELoss()
        self.celoss = torch.nn.CrossEntropyLoss()

    # def loss(self, pred, true_label):
    #     return self.bceloss(pred, true_label)

class GPKG_EMBEDD(BaseModel):
    def __init__(self, edge_index, edge_type, num_rel, params=None):
        super(GPKG_EMBEDD, self).__init__(params)

        self.edge_index = edge_index
        self.edge_type = edge_type
        self.p.gcn_dim = self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim
        self.init_embed = get_param((self.p.num_ent, self.p.init_dim))
        self.device = self.edge_index.device
        self.gamma = 9.0

        # if self.p.num_bases > 0:
        #     self.init_rel = get_param((self.p.num_bases, self.p.init_dim))
        # else:
        #     self.init_rel = get_param((num_rel * 2, self.p.init_dim))
        self.init_rel = get_param((num_rel * 2, self.p.init_dim))
        if self.p.num_bases > 0:
            self.conv1 = CompGATv3(in_channels=self.p.init_dim, out_channels=self.p.gcn_dim, rel_dim=self.p.rel_dim, 
                                   drop=self.p.encoder_drop, bias=True, op=self.p.op, beta=self.p.beta)
            # self.conv1 = GPKGConvBasis(self.p.init_dim, self.p.gcn_dim, num_rel, self.p.num_bases, act=self.act,
            #                            params=self.p)
            # self.conv2 = GPKGConv(self.p.gcn_dim, self.p.embed_dim, num_rel, act=self.act,
            #                       params=self.p) if self.p.gcn_layer == 2 else None
        elif self.p.num_bases == -3:
            self.conv1 = GPKGConv(self.p.init_dim, self.p.gcn_dim, num_rel, act=self.act, params=self.p)
            self.conv2 = CompGATv3(in_channels=self.p.gcn_dim, out_channels=self.p.embed_dim, rel_dim=self.p.rel_dim, 
                                   drop=self.p.encoder_drop, bias=True, op=self.p.op, beta=self.p.beta)if self.p.gcn_layer == 2 else None
            
        elif self.p.num_bases == -2:
            self.conv1 = CompGATv3(in_channels=self.p.init_dim, out_channels=self.p.embed_dim, rel_dim=self.p.rel_dim, 
                                   drop=self.p.encoder_drop, bias=True, op=self.p.op, beta=self.p.beta)
            self.conv2 = GPKGConv(self.p.embed_dim, self.p.gcn_dim, num_rel, act=self.act, params=self.p)if self.p.gcn_layer == 2 else None
            
        elif self.p.num_bases == -1:
            self.conv1 = GPKGConv(self.p.init_dim, self.p.gcn_dim, num_rel, act=self.act, params=self.p)
            self.conv2 = CompGAT(in_channels=self.p.gcn_dim, out_channels=self.p.embed_dim, rel_dim=self.p.rel_dim, 
                                   drop=self.p.encoder_drop, bias=True, op=self.p.op)if self.p.gcn_layer == 2 else None
        elif self.p.num_bases == -4:
            self.conv1 = Transformer(in_channels=self.p.init_dim, out_channels=self.p.gcn_dim, rel_dim=self.p.rel_dim, drop=self.p.encoder_drop, bias=True, op=self.p.op, beta=self.p.beta, num_head=1, final_layer=False)
        
        elif self.p.num_bases == -5:
            self.conv1 = GPKGConv(self.p.init_dim, self.p.gcn_dim, num_rel, act=self.act, params=self.p)
            self.conv2 = Transformer(in_channels=self.p.gcn_dim, out_channels=self.p.embed_dim, rel_dim=self.p.rel_dim, drop=self.p.encoder_drop, bias=True, op=self.p.op, beta=self.p.beta, num_head=1, final_layer=False)if self.p.gcn_layer == 2 else None
        else:
            self.conv1 = GPKGConv(self.p.init_dim, self.p.gcn_dim, num_rel, act=self.act, params=self.p)
            self.conv2 = GPKGConv(self.p.gcn_dim, self.p.embed_dim, num_rel, act=self.act,
                                  params=self.p) if self.p.gcn_layer == 2 else None

        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))

    #def forward_embedd(self, sub, rel, obj, drop1, drop2):
    def forward_embedd(self, sub, rel, drop1, drop2):

        r = self.init_rel
        #print(self.init_embed.shape, r.shape)
        #print(self.edge_index.shape)
        x, r = self.conv1(self.init_embed, self.edge_index, self.edge_type, rel_embed=r)
        #x, r = self.conv1(self.init_embed, self.edge_index, self.edge_type, rel_embed=r)
        x = drop1(x)
        #print(x.shape)
        #print(x.shape, r.shape)
        #print(self.edge_index.shape)
        x, r = self.conv2(x, self.edge_index, self.edge_type, rel_emb=r) if self.p.gcn_layer == 2 else (x, r)
        x = drop2(x) if self.p.gcn_layer == 2 else x
        #print(x.shape)
        #print(x.shape)
        sub_emb = torch.index_select(x, 0, sub)
        rel_emb = torch.index_select(r, 0, rel)
        #print(sub_emb.shape, rel_emb.shape)
        #obj_emb = torch.index_select(x, 0, obj)

        #return sub_emb, rel_emb, obj_emb, x
        return sub_emb, rel_emb, x
    



    def loss_function(y, t, drop_rate):
        loss = F.binary_cross_entropy_with_logits(y, t, reduce=False)# 这个loss全是负数！

        loss_mul = loss * t # 只包含正样本的损失，负样本损失为0，正样本损失均为负数
        ind_sorted = np.argsort(loss_mul.cpu().data).cuda() # 将正样本的损失进行排序，默认从小到大，返回索引
        print(ind_sorted.shape)
        loss_sorted = loss[ind_sorted]

        remember_rate = 1 - drop_rate
        num_remember = int(remember_rate * len(loss_sorted))

        ind_update = ind_sorted[:num_remember]

        loss_update = F.binary_cross_entropy_with_logits(y[ind_update], t[ind_update])

        return loss_update


class GPKG_PREDICT(GPKG_EMBEDD):
    def __init__(self, edge_index, edge_type, chequer_perm, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)

        self.inp_drop = torch.nn.Dropout(self.p.inp_drop)
        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.p.hid_drop2)
        self.feature_map_drop = torch.nn.Dropout2d(self.p.feat_drop)
        self.bn0 = torch.nn.BatchNorm2d(self.p.perm)
        #self.supconloss = SupConLoss(temperature=self.p.temp1, contrast_mode="all", base_temperature=self.p.temp1).to(torch.device("cuda"))

        flat_sz_h = self.p.k_h
        flat_sz_w = 2 * self.p.k_w
        self.padding = 0

        self.bn1 = torch.nn.BatchNorm2d(self.p.num_filt * self.p.perm)
        self.flat_sz = flat_sz_h * flat_sz_w * self.p.num_filt * self.p.perm

        self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)
        self.fc = torch.nn.Linear(self.flat_sz, self.p.embed_dim)
        self.chequer_perm = chequer_perm

        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))
        self.register_parameter('conv_filt', Parameter(torch.zeros(self.p.num_filt, 1, self.p.ker_sz, self.p.ker_sz)));
        xavier_normal_(self.conv_filt)
        self.h_ops_dict = nn.ModuleDict({
            'p': nn.Linear(200, self.p.gcn_dim, bias=False),
            'b': nn.BatchNorm1d(self.p.gcn_dim),
            'd': nn.Dropout(self.p.hid_drop),
            'a': nn.Tanh(),
        })

        self.t_ops_dict = nn.ModuleDict({
            'p': nn.Linear(200, self.p.gcn_dim, bias=False),
            'b': nn.BatchNorm1d(self.p.gcn_dim),
            'd': nn.Dropout(self.p.hid_drop),
            'a': nn.Tanh(),
        })

        self.r_ops_dict = nn.ModuleDict({
            'p': nn.Linear(200, self.p.gcn_dim, bias=False),
            'b': nn.BatchNorm1d(self.p.gcn_dim),
            'd': nn.Dropout(self.p.hid_drop),
            'a': nn.Tanh(),
        })

    def loss(self, pred, true_label=None, sub_samp=None):
        label_pos = true_label[0];
        label_neg = true_label[1:]
        loss = self.bceloss(pred, true_label)
        #loss = self.celoss(pred, true_label)
        #loss = self.loss_function(pred, true_label, 0.2)
        return loss

    def circular_padding_chw(self, batch, padding):
        upper_pad = batch[..., -padding:, :]
        lower_pad = batch[..., :padding, :]
        temp = torch.cat([upper_pad, batch, lower_pad], dim=2)

        left_pad = temp[..., -padding:]
        right_pad = temp[..., :padding]
        padded = torch.cat([left_pad, temp, right_pad], dim=3)
        return padded
    
    def exop(self, x, r, x_ops=None, r_ops=None, diff_ht=False):
        x_head = x_tail = x
        if len(x_ops) > 0:
            for x_op in x_ops.split("."):
                #if diff_ht:
                    x_head = self.h_ops_dict[x_op](x_head)
                    #x_tail = self.t_ops_dict[x_op](x_tail)
                # else:
                #     x_head = x_tail = self.h_ops_dict[x_op](x_head)

        if len(r_ops) > 0:
            for r_op in r_ops.split("."):
                r = self.r_ops_dict[r_op](r)

        return x_head, r

    #def forward(self, sub, rel, obj, neg_ents):
    def forward(self, sub, rel, neg_ents):
        #sub_emb, rel_emb, obj_emb, all_ent = self.forward_embedd(sub, rel, obj, self.hidden_drop, self.feature_map_drop)
        sub_emb, rel_emb, all_ent = self.forward_embedd(sub, rel, self.hidden_drop, self.feature_map_drop)
        #print(sub_emb.shape, rel_emb.shape)
        # x_h, x_t, r = self.exop(sub_emb, rel_emb, 'p.b.a', 'p.b.a')
        # obj_emb = x_h * r
        # x = torch.mm(obj_emb, all_ent.transpose(1, 0))
        # x += self.bias.expand_as(x)
        #sub_emb, rel_emb = self.exop(sub_emb, rel_emb ,'p.b.a','p.b.a')
        
        

        comb_emb = torch.cat([sub_emb, rel_emb], dim=1)
        chequer_perm = comb_emb[:, self.chequer_perm]
        stack_inp = chequer_perm.reshape((-1, self.p.perm, 2 * self.p.k_w, self.p.k_h))
        #print(stack_inp.shape)
        x = stack_inp = self.bn0(stack_inp)
        x = self.circular_padding_chw(x, self.p.ker_sz // 2)
        x = F.conv2d(x, self.conv_filt.repeat(self.p.perm, 1, 1, 1), padding=self.padding, groups=self.p.perm)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        cl_x = x
        #print(x.shape)
        x = F.relu(x)
        x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)

        # obj_emb = sub_emb + rel_emb

        # x = self.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)
        # score = torch.sigmoid(x)
        
        #return cl_x, x, pred, obj_emb
        pred	= torch.sigmoid(x)
        return pred
        #return x
        #return score

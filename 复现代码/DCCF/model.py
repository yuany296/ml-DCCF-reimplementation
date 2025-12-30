import torch
import numpy as np
import torch_sparse
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F

class DCCF(nn.Module):
    def __init__(self, data_config, args):
        super(DCCF, self).__init__()

        self.n_users = data_config['n_users']  #用户数
        self.n_items = data_config['n_items']  #物品数

        self.plain_adj = data_config['plain_adj'] #邻接矩阵
        self.all_h_list = data_config['all_h_list'] #邻接矩阵行索引
        self.all_t_list = data_config['all_t_list'] #邻接矩阵列索引
        self.A_in_shape = self.plain_adj.tocoo().shape #邻接矩阵形状
        
        # 修改1: 设备自动选择
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 修改2: 使用 self.device 而不是写死的 .cuda()
        self.A_indices = torch.tensor([self.all_h_list, self.all_t_list], dtype=torch.long).to(self.device) #邻接矩阵的行列索引张量
        self.D_indices = torch.tensor([list(range(self.n_users + self.n_items)), list(range(self.n_users + self.n_items))], dtype=torch.long).to(self.device) #度矩阵的行列索引张量
        self.all_h_list = torch.LongTensor(self.all_h_list).to(self.device) #邻接矩阵行索引张量
        self.all_t_list = torch.LongTensor(self.all_t_list).to(self.device) #邻接矩阵列索引张量
        self.G_indices, self.G_values = self._cal_sparse_adj() #归一化邻接矩阵的行列索引和值张量

        self.emb_dim = args.embed_size #嵌入维度
        self.n_layers = args.n_layers #图卷积层数
        self.n_intents = args.n_intents #意图数
        self.temp = args.temp #温度系数

        self.batch_size = args.batch_size #批量大小
        self.emb_reg = args.emb_reg #嵌入正则化系数
        self.cen_reg = args.cen_reg #意图原型正则化系数
        self.ssl_reg = args.ssl_reg #自监督学习正则化系数

        """
        *********************************************************
        Create Model Parameters 创建模型参数
        """
        self.user_embedding = nn.Embedding(self.n_users, self.emb_dim) #用户嵌入矩阵
        self.item_embedding = nn.Embedding(self.n_items, self.emb_dim) #物品嵌入矩阵

        _user_intent = torch.empty(self.emb_dim, self.n_intents) #用户意图原型矩阵
        nn.init.xavier_normal_(_user_intent) #初始化用户意图原型矩阵
        self.user_intent = torch.nn.Parameter(_user_intent, requires_grad=True) #将用户意图原型矩阵注册为模型参数
        _item_intent = torch.empty(self.emb_dim, self.n_intents) #物品意图原型矩阵
        nn.init.xavier_normal_(_item_intent) #初始化物品意图原型矩阵
        self.item_intent = torch.nn.Parameter(_item_intent, requires_grad=True) #将物品意图原型矩阵注册为模型参数

        """
        *********************************************************
        Initialize Weights 初始化权重
        """
        self._init_weight()

    def _init_weight(self):
        '''初始化模型参数'''
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

    def _cal_sparse_adj(self):
        '''计算归一化邻接矩阵'''
        # 修改3: 使用 self.device
        A_values = torch.ones(size=(len(self.all_h_list), 1)).view(-1).to(self.device)#邻接矩阵的值张量

        A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list, value=A_values, sparse_sizes=self.A_in_shape).to(self.device)#邻接矩阵的稀疏张量表示
        D_values = A_tensor.sum(dim=1).pow(-0.5) #度矩阵的值张量

        G_indices, G_values = torch_sparse.spspmm(self.D_indices, D_values, self.A_indices, A_values, self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])#归一化邻接矩阵的行列索引和值张量
        G_indices, G_values = torch_sparse.spspmm(G_indices, G_values, self.D_indices, D_values, self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])#归一化邻接矩阵的行列索引和值张量

        return G_indices, G_values

    def _adaptive_mask(self, head_embeddings, tail_embeddings):
        '''
        计算自适应掩码
        '''
        head_embeddings = torch.nn.functional.normalize(head_embeddings) #归一化头节点嵌入
        tail_embeddings = torch.nn.functional.normalize(tail_embeddings)    #归一化尾节点嵌入
        edge_alpha = (torch.sum(head_embeddings * tail_embeddings, dim=1).view(-1) + 1) / 2 #计算边权重

        # 修改4: 使用 self.device
        A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list, value=edge_alpha, sparse_sizes=self.A_in_shape).to(self.device)
        D_scores_inv = A_tensor.sum(dim=1).pow(-1).nan_to_num(0, 0, 0).view(-1)

        G_indices = torch.stack([self.all_h_list, self.all_t_list], dim=0)
        G_values = D_scores_inv[self.all_h_list] * edge_alpha

        return G_indices, G_values

    def inference(self):
        all_embeddings = [torch.concat([self.user_embedding.weight, self.item_embedding.weight], dim=0)]

        gnn_embeddings = []
        int_embeddings = []
        gaa_embeddings = []
        iaa_embeddings = []

        for i in range(0, self.n_layers):
            # Graph-based Message Passing
            gnn_layer_embeddings = torch_sparse.spmm(self.G_indices, self.G_values, self.A_in_shape[0], self.A_in_shape[1], all_embeddings[i])

            # Intent-aware Information Aggregation
            u_embeddings, i_embeddings = torch.split(all_embeddings[i], [self.n_users, self.n_items], 0)
            u_int_embeddings = torch.softmax(u_embeddings @ self.user_intent, dim=1) @ self.user_intent.T
            i_int_embeddings = torch.softmax(i_embeddings @ self.item_intent, dim=1) @ self.item_intent.T
            int_layer_embeddings = torch.concat([u_int_embeddings, i_int_embeddings], dim=0)

            # Adaptive Augmentation
            gnn_head_embeddings = torch.index_select(gnn_layer_embeddings, 0, self.all_h_list)
            gnn_tail_embeddings = torch.index_select(gnn_layer_embeddings, 0, self.all_t_list)
            int_head_embeddings = torch.index_select(int_layer_embeddings, 0, self.all_h_list)
            int_tail_embeddings = torch.index_select(int_layer_embeddings, 0, self.all_t_list)
            G_graph_indices, G_graph_values = self._adaptive_mask(gnn_head_embeddings, gnn_tail_embeddings)
            G_inten_indices, G_inten_values = self._adaptive_mask(int_head_embeddings, int_tail_embeddings)

            gaa_layer_embeddings = torch_sparse.spmm(G_graph_indices, G_graph_values, self.A_in_shape[0], self.A_in_shape[1], all_embeddings[i])
            iaa_layer_embeddings = torch_sparse.spmm(G_inten_indices, G_inten_values, self.A_in_shape[0], self.A_in_shape[1], all_embeddings[i])

            gnn_embeddings.append(gnn_layer_embeddings)
            int_embeddings.append(int_layer_embeddings)
            gaa_embeddings.append(gaa_layer_embeddings)
            iaa_embeddings.append(iaa_layer_embeddings)

            all_embeddings.append(gnn_layer_embeddings + int_layer_embeddings + gaa_layer_embeddings + iaa_layer_embeddings + all_embeddings[i])

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.sum(all_embeddings, dim=1, keepdim=False)

        self.ua_embedding, self.ia_embedding = torch.split(all_embeddings, [self.n_users, self.n_items], 0)

        return gnn_embeddings, int_embeddings, gaa_embeddings, iaa_embeddings

    def cal_ssl_loss(self, users, items, gnn_emb, int_emb, gaa_emb, iaa_emb):
        users = torch.unique(users)
        items = torch.unique(items)

        cl_loss = 0.0

        def cal_loss(emb1, emb2):
            pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.temp)
            neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / self.temp), axis=1)
            loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
            loss /= pos_score.shape[0]
            return loss

        for i in range(len(gnn_emb)):
            u_gnn_embs, i_gnn_embs = torch.split(gnn_emb[i], [self.n_users, self.n_items], 0)
            u_int_embs, i_int_embs = torch.split(int_emb[i], [self.n_users, self.n_items], 0)
            u_gaa_embs, i_gaa_embs = torch.split(gaa_emb[i], [self.n_users, self.n_items], 0)
            u_iaa_embs, i_iaa_embs = torch.split(iaa_emb[i], [self.n_users, self.n_items], 0)

            u_gnn_embs = F.normalize(u_gnn_embs[users], dim=1)
            u_int_embs = F.normalize(u_int_embs[users], dim=1)
            u_gaa_embs = F.normalize(u_gaa_embs[users], dim=1)
            u_iaa_embs = F.normalize(u_iaa_embs[users], dim=1)

            i_gnn_embs = F.normalize(i_gnn_embs[items], dim=1)
            i_int_embs = F.normalize(i_int_embs[items], dim=1)
            i_gaa_embs = F.normalize(i_gaa_embs[items], dim=1)
            i_iaa_embs = F.normalize(i_iaa_embs[items], dim=1)

            cl_loss += cal_loss(u_gnn_embs, u_int_embs)
            cl_loss += cal_loss(u_gnn_embs, u_gaa_embs)
            cl_loss += cal_loss(u_gnn_embs, u_iaa_embs)

            cl_loss += cal_loss(i_gnn_embs, i_int_embs)
            cl_loss += cal_loss(i_gnn_embs, i_gaa_embs)
            cl_loss += cal_loss(i_gnn_embs, i_iaa_embs)

        return cl_loss

    def forward(self, users, pos_items, neg_items):
        # 修改5: 使用 self.device
        users = torch.LongTensor(users).to(self.device)
        pos_items = torch.LongTensor(pos_items).to(self.device)
        neg_items = torch.LongTensor(neg_items).to(self.device)

        gnn_embeddings, int_embeddings, gaa_embeddings, iaa_embeddings = self.inference()

        # bpr
        u_embeddings = self.ua_embedding[users]
        pos_embeddings = self.ia_embedding[pos_items]
        neg_embeddings = self.ia_embedding[neg_items]
        pos_scores = torch.sum(u_embeddings * pos_embeddings, 1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, 1)
        mf_loss = torch.mean(F.softplus(neg_scores - pos_scores))

        # embeddings
        u_embeddings_pre = self.user_embedding(users)
        pos_embeddings_pre = self.item_embedding(pos_items)
        neg_embeddings_pre = self.item_embedding(neg_items)
        emb_loss = (u_embeddings_pre.norm(2).pow(2) + pos_embeddings_pre.norm(2).pow(2) + neg_embeddings_pre.norm(2).pow(2))
        emb_loss = self.emb_reg * emb_loss

        # intent prototypes
        cen_loss = (self.user_intent.norm(2).pow(2) + self.item_intent.norm(2).pow(2))
        cen_loss = self.cen_reg * cen_loss

        # self-supervise learning
        cl_loss = self.ssl_reg * self.cal_ssl_loss(users, pos_items, gnn_embeddings, int_embeddings, gaa_embeddings, iaa_embeddings)

        return mf_loss, emb_loss, cen_loss, cl_loss

    def predict(self, users):
        # 修改6: 使用 self.device
        u_embeddings = self.ua_embedding[torch.LongTensor(users).to(self.device)]
        i_embeddings = self.ia_embedding
        batch_ratings = torch.matmul(u_embeddings, i_embeddings.T)
        return batch_ratings
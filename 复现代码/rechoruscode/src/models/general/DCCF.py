# -*- coding: UTF-8 -*-
import numpy as np
import torch
import torch_sparse
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F

from models.BaseModel import GeneralModel

class DCCF(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'n_layers', 'n_intents', 'temp', 'emb_reg', 'cen_reg', 'ssl_reg']
    
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--n_layers', type=int, default=3,
                            help='Number of graph convolution layers.')
        parser.add_argument('--n_intents', type=int, default=8,
                            help='Number of intents for intent-aware aggregation.')
        parser.add_argument('--temp', type=float, default=0.1,
                            help='Temperature parameter for SSL loss.')
        parser.add_argument('--emb_reg', type=float, default=1e-4,
                            help='Regularization coefficient for embeddings.')
        parser.add_argument('--cen_reg', type=float, default=1e-4,
                            help='Regularization coefficient for intent centroids.')
        parser.add_argument('--ssl_reg', type=float, default=0.1,
                            help='Regularization coefficient for self-supervised learning loss.')
        return GeneralModel.parse_model_args(parser)
    
    @staticmethod
    def build_adjmat(user_count, item_count, train_mat, selfloop_flag=False):
        """Build adjacency matrix from training data"""
        R = sp.dok_matrix((user_count, item_count), dtype=np.float32)
        for user in train_mat:
            for item in train_mat[user]:
                R[user, item] = 1
        R = R.tolil()
        
        adj_mat = sp.dok_matrix((user_count + item_count, user_count + item_count), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        
        adj_mat[:user_count, user_count:] = R
        adj_mat[user_count:, :user_count] = R.T
        adj_mat = adj_mat.todok()
        
        if selfloop_flag:
            adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
        
        return adj_mat.tocsr()
    
    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.n_layers = args.n_layers
        self.n_intents = args.n_intents
        self.temp = args.temp
        self.emb_reg = args.emb_reg
        self.cen_reg = args.cen_reg
        self.ssl_reg = args.ssl_reg
        
        # Build adjacency matrix
        self.plain_adj = self.build_adjmat(
            corpus.n_users, 
            corpus.n_items, 
            corpus.train_clicked_set, 
            selfloop_flag=False
        )
        
        # Convert sparse matrix to indices
        plain_adj_coo = self.plain_adj.tocoo()
        self.all_h_list = plain_adj_coo.row.tolist()
        self.all_t_list = plain_adj_coo.col.tolist()
        self.A_in_shape = self.plain_adj.shape
        
        # Define model parameters
        self._define_params()
        self.apply(self.init_weights)
        
        # Cache for embeddings
        self._cached_embeddings = None
        self._cached_final_embeddings = None

        self._embeddings_computed = False
        self._compute_embeddings_at_init = False

    def train(self, mode=True):
        """切换到训练模式"""
        super().train(mode)
        #if mode:
            # 清空缓存，让forward重新计算
            #self._cached_embeddings = None
            #self._cached_final_embeddings = None
    
    def _define_params(self):
        """Define model parameters"""
        # User and item embeddings
        self.user_embedding = nn.Embedding(self.user_num, self.emb_size)
        self.item_embedding = nn.Embedding(self.item_num, self.emb_size)
        
        # Intent prototypes
        _user_intent = torch.empty(self.emb_size, self.n_intents)
        nn.init.xavier_normal_(_user_intent)
        self.user_intent = nn.Parameter(_user_intent, requires_grad=True)
        
        _item_intent = torch.empty(self.emb_size, self.n_intents)
        nn.init.xavier_normal_(_item_intent)
        self.item_intent = nn.Parameter(_item_intent, requires_grad=True)
        
        # Prepare sparse indices
        self.register_buffer('A_indices', torch.tensor([self.all_h_list, self.all_t_list], dtype=torch.long))
        self.register_buffer('D_indices', torch.tensor(
            [list(range(self.user_num + self.item_num)), list(range(self.user_num + self.item_num))], 
            dtype=torch.long
        ))
        self.register_buffer('all_h_list_tensor', torch.LongTensor(self.all_h_list))
        self.register_buffer('all_t_list_tensor', torch.LongTensor(self.all_t_list))
        
        # These will be initialized when moved to device
        self.G_indices = None
        self.G_values = None
    
    def _cal_sparse_adj(self):
        """Calculate normalized adjacency matrix"""
        device = self.A_indices.device
        A_values = torch.ones(size=(len(self.all_h_list), 1)).view(-1).to(device)
        
        A_tensor = torch_sparse.SparseTensor(
            row=self.all_h_list_tensor, 
            col=self.all_t_list_tensor, 
            value=A_values, 
            sparse_sizes=self.A_in_shape
        ).to(device)
        
        D_values = A_tensor.sum(dim=1).pow(-0.5)
        
        G_indices, G_values = torch_sparse.spspmm(
            self.D_indices, D_values, 
            self.A_indices, A_values, 
            self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1]
        )
        
        G_indices, G_values = torch_sparse.spspmm(
            G_indices, G_values, 
            self.D_indices, D_values, 
            self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1]
        )
        
        return G_indices, G_values
    
    def _adaptive_mask(self, head_embeddings, tail_embeddings):
        """Create adaptive mask based on node similarities"""
        head_embeddings = F.normalize(head_embeddings, dim=1)
        tail_embeddings = F.normalize(tail_embeddings, dim=1)
        edge_alpha = (torch.sum(head_embeddings * tail_embeddings, dim=1).view(-1) + 1) / 2
        
        A_tensor = torch_sparse.SparseTensor(
            row=self.all_h_list_tensor, 
            col=self.all_t_list_tensor, 
            value=edge_alpha, 
            sparse_sizes=self.A_in_shape
        ).to(head_embeddings.device)
        
        D_scores_inv = A_tensor.sum(dim=1).pow(-1).nan_to_num(0, 0, 0).view(-1)
        
        G_indices = torch.stack([self.all_h_list_tensor, self.all_t_list_tensor], dim=0)
        G_values = D_scores_inv[self.all_h_list_tensor] * edge_alpha
        
        return G_indices, G_values
    
    def inference(self, feed_dict=None):
        """处理BaseRunner的调用"""
        if feed_dict is None:
            return None
    
        # 根据训练/评估模式返回不同结果
        if self.training:  # 训练模式
            return self.forward(feed_dict)
        else:  # 评估模式
            return self.predict(feed_dict)

    
    
    def cal_ssl_loss(self, users, items, gnn_emb, int_emb, gaa_emb, iaa_emb):
        """Calculate self-supervised learning loss"""
        # Ensure indices are within bounds
        users = torch.clamp(users, 0, self.user_num - 1)
        items = torch.clamp(items, 0, self.item_num - 1)
        
        users = torch.unique(users)
        items = torch.unique(items)
        
        # Check for empty tensors
        if len(users) == 0 or len(items) == 0:
            return torch.tensor(0.0, device=users.device)
        
        cl_loss = 0.0
        
        def cal_loss(emb1, emb2):
            emb1 = F.normalize(emb1, dim=1)
            emb2 = F.normalize(emb2, dim=1)
            pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.temp)
            neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / self.temp), dim=1)
            loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
            loss /= pos_score.shape[0]
            return loss
        
        for i in range(len(gnn_emb)):
            u_gnn_embs, i_gnn_embs = torch.split(gnn_emb[i], [self.user_num, self.item_num], 0)
            u_int_embs, i_int_embs = torch.split(int_emb[i], [self.user_num, self.item_num], 0)
            u_gaa_embs, i_gaa_embs = torch.split(gaa_emb[i], [self.user_num, self.item_num], 0)
            u_iaa_embs, i_iaa_embs = torch.split(iaa_emb[i], [self.user_num, self.item_num], 0)
            
            # User contrastive losses
            cl_loss += cal_loss(u_gnn_embs[users], u_int_embs[users])
            cl_loss += cal_loss(u_gnn_embs[users], u_gaa_embs[users])
            cl_loss += cal_loss(u_gnn_embs[users], u_iaa_embs[users])
            
            # Item contrastive losses
            cl_loss += cal_loss(i_gnn_embs[items], i_int_embs[items])
            cl_loss += cal_loss(i_gnn_embs[items], i_gaa_embs[items])
            cl_loss += cal_loss(i_gnn_embs[items], i_iaa_embs[items])
        
        return cl_loss
    
    
    def forward(self, feed_dict):
        """
        Forward pass for training
        According to GeneralModel, feed_dict contains:
        - user_id: [batch_size]
        - item_id: [batch_size, 1+num_neg] where first column is positive item
        """
        # Get batch data
        user_ids = feed_dict['user_id'].to(self.device)
        item_ids = feed_dict['item_id'].to(self.device)  # shape: [batch_size, 1+num_neg]
        
        # ========== 计算嵌入 ==========
        # 初始化稀疏邻接矩阵
        if self.G_indices is None:
            self.G_indices, self.G_values = self._cal_sparse_adj()
        
        all_embeddings = [torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)]
        
        gnn_embeddings = []
        int_embeddings = []
        gaa_embeddings = []
        iaa_embeddings = []
        
        for i in range(self.n_layers):
            # Graph-based Message Passing
            gnn_layer_embeddings = torch_sparse.spmm(
                self.G_indices, self.G_values, 
                self.A_in_shape[0], self.A_in_shape[1], 
                all_embeddings[i]
            )
            
            # Intent-aware Information Aggregation
            u_embeddings, i_embeddings = torch.split(
                all_embeddings[i], [self.user_num, self.item_num], 0
            )
            u_int_embeddings = torch.softmax(u_embeddings @ self.user_intent, dim=1) @ self.user_intent.T
            i_int_embeddings = torch.softmax(i_embeddings @ self.item_intent, dim=1) @ self.item_intent.T
            int_layer_embeddings = torch.cat([u_int_embeddings, i_int_embeddings], dim=0)
            
            # Adaptive Augmentation
            gnn_head_embeddings = torch.index_select(gnn_layer_embeddings, 0, self.all_h_list_tensor)
            gnn_tail_embeddings = torch.index_select(gnn_layer_embeddings, 0, self.all_t_list_tensor)
            int_head_embeddings = torch.index_select(int_layer_embeddings, 0, self.all_h_list_tensor)
            int_tail_embeddings = torch.index_select(int_layer_embeddings, 0, self.all_t_list_tensor)
            
            G_graph_indices, G_graph_values = self._adaptive_mask(gnn_head_embeddings, gnn_tail_embeddings)
            G_inten_indices, G_inten_values = self._adaptive_mask(int_head_embeddings, int_tail_embeddings)
            
            gaa_layer_embeddings = torch_sparse.spmm(
                G_graph_indices, G_graph_values, 
                self.A_in_shape[0], self.A_in_shape[1], 
                all_embeddings[i]
            )
            iaa_layer_embeddings = torch_sparse.spmm(
                G_inten_indices, G_inten_values, 
                self.A_in_shape[0], self.A_in_shape[1], 
                all_embeddings[i]
            )
            
            gnn_embeddings.append(gnn_layer_embeddings)
            int_embeddings.append(int_layer_embeddings)
            gaa_embeddings.append(gaa_layer_embeddings)
            iaa_embeddings.append(iaa_layer_embeddings)
            
            # Combine all embeddings
            combined = (gnn_layer_embeddings + int_layer_embeddings + 
                    gaa_layer_embeddings + iaa_layer_embeddings + all_embeddings[i])
            all_embeddings.append(combined)
        
        # Sum all layer embeddings
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.sum(all_embeddings, dim=1, keepdim=False)
        
        # 分割用户和物品嵌入
        ua_embedding, ia_embedding = torch.split(all_embeddings, [self.user_num, self.item_num], 0)
        # ========== 嵌入计算结束 ==========
        
        # Get predictions for all items (for BPR loss)
        u_embeddings = ua_embedding[user_ids]  # [batch_size, emb_size]
        i_embeddings = ia_embedding[item_ids]  # [batch_size, 1+num_neg, emb_size]
        
        # Calculate scores
        scores = torch.sum(u_embeddings.unsqueeze(1) * i_embeddings, dim=-1)  # [batch_size, 1+num_neg]
        
        # Use parent class's BPR loss
        predictions = scores
        out_dict = {'prediction': predictions}
        
        # Calculate DCCF-specific losses
        # 1. Embedding regularization
        u_embeddings_pre = self.user_embedding(user_ids)
        pos_embeddings_pre = self.item_embedding(item_ids[:, 0])
        emb_loss = self.emb_reg * (
            u_embeddings_pre.norm(2).pow(2) + 
            pos_embeddings_pre.norm(2).pow(2)
        )
        
        # 2. Intent centroid regularization
        cen_loss = self.cen_reg * (
            self.user_intent.norm(2).pow(2) + 
            self.item_intent.norm(2).pow(2)
        )
        
        # 3. Self-supervised learning loss
        # Get all items in batch (positive + negatives)
        batch_items = item_ids.flatten().unique()
        cl_loss = self.ssl_reg * self.cal_ssl_loss(
            user_ids, batch_items, 
            gnn_embeddings, int_embeddings, gaa_embeddings, iaa_embeddings
        )
        
        # Store losses for monitoring
        #self.check_list.extend([
            #('emb_loss', emb_loss.item()),
            #('cen_loss', cen_loss.item()),
            #('cl_loss', cl_loss.item())
        #])
        
        # Calculate total loss = BPR loss + DCCF losses
        bpr_loss = super().loss(out_dict)  # Use parent's BPR loss
        total_loss = bpr_loss + emb_loss + cen_loss + cl_loss
        
        # Update out_dict with total loss
        out_dict['loss'] = total_loss
        
        return out_dict
    
    def predict(self, feed_dict):
        """Predict for evaluation"""
        user_ids = feed_dict['user_id'].to(self.device)
        item_ids = feed_dict['item_id'].to(self.device)
        # ========== 计算嵌入 ==========
        # 初始化稀疏邻接矩阵
        if self.G_indices is None:
            self.G_indices, self.G_values = self._cal_sparse_adj()
        
        all_embeddings = [torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)]
        
        gnn_embeddings = []
        int_embeddings = []
        gaa_embeddings = []
        iaa_embeddings = []
        
        for i in range(self.n_layers):
            # Graph-based Message Passing
            gnn_layer_embeddings = torch_sparse.spmm(
                self.G_indices, self.G_values, 
                self.A_in_shape[0], self.A_in_shape[1], 
                all_embeddings[i]
            )
            
            # Intent-aware Information Aggregation
            u_embeddings, i_embeddings = torch.split(
                all_embeddings[i], [self.user_num, self.item_num], 0
            )
            u_int_embeddings = torch.softmax(u_embeddings @ self.user_intent, dim=1) @ self.user_intent.T
            i_int_embeddings = torch.softmax(i_embeddings @ self.item_intent, dim=1) @ self.item_intent.T
            int_layer_embeddings = torch.cat([u_int_embeddings, i_int_embeddings], dim=0)
            
            # Adaptive Augmentation
            gnn_head_embeddings = torch.index_select(gnn_layer_embeddings, 0, self.all_h_list_tensor)
            gnn_tail_embeddings = torch.index_select(gnn_layer_embeddings, 0, self.all_t_list_tensor)
            int_head_embeddings = torch.index_select(int_layer_embeddings, 0, self.all_h_list_tensor)
            int_tail_embeddings = torch.index_select(int_layer_embeddings, 0, self.all_t_list_tensor)
            
            G_graph_indices, G_graph_values = self._adaptive_mask(gnn_head_embeddings, gnn_tail_embeddings)
            G_inten_indices, G_inten_values = self._adaptive_mask(int_head_embeddings, int_tail_embeddings)
            
            gaa_layer_embeddings = torch_sparse.spmm(
                G_graph_indices, G_graph_values, 
                self.A_in_shape[0], self.A_in_shape[1], 
                all_embeddings[i]
            )
            iaa_layer_embeddings = torch_sparse.spmm(
                G_inten_indices, G_inten_values, 
                self.A_in_shape[0], self.A_in_shape[1], 
                all_embeddings[i]
            )
            
            gnn_embeddings.append(gnn_layer_embeddings)
            int_embeddings.append(int_layer_embeddings)
            gaa_embeddings.append(gaa_layer_embeddings)
            iaa_embeddings.append(iaa_layer_embeddings)
            
            # Combine all embeddings
            combined = (gnn_layer_embeddings + int_layer_embeddings + 
                    gaa_layer_embeddings + iaa_layer_embeddings + all_embeddings[i])
            all_embeddings.append(combined)
        
        # Sum all layer embeddings
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.sum(all_embeddings, dim=1, keepdim=False)
        
        # 分割用户和物品嵌入
        ua_embedding, ia_embedding = torch.split(all_embeddings, [self.user_num, self.item_num], 0)

        # ========== 嵌入计算结束 ==========
        
        # Get user embeddings
        u_embeddings = ua_embedding[user_ids]  # [batch_size, emb_size]
        
        # If test_all mode, predict for all items
        if self.test_all:
            i_embeddings = ia_embedding  # [item_num, emb_size]
            predictions = torch.matmul(u_embeddings, i_embeddings.T)  # [batch_size, item_num]
        else:
            # Predict for specific items
            i_embeddings = ia_embedding[item_ids]  # [batch_size, n_candidates, emb_size]
            predictions = torch.sum(u_embeddings.unsqueeze(1) * i_embeddings, dim=-1)  # [batch_size, n_candidates]
        
        return {'prediction': predictions}
    
    
    def eval(self):
        """Switch to evaluation mode"""
        super().eval()
        # Keep cache for faster evaluation
    
    def actions_after_train(self):
        """Clear cache after training"""
        self._cached_embeddings = None
        self._cached_final_embeddings = None
    
    def to(self, device):
        """Move model to device and initialize sparse adjacency"""
        moved = super().to(device)
    
        # 初始化稀疏邻接矩阵
        if self.G_indices is None:
            self.G_indices, self.G_values = self._cal_sparse_adj()
        
        
        return moved
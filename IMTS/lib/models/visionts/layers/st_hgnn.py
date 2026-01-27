import torch
import torch.nn as nn
import torch.nn.functional as F

class ST_HGNN_Layer(nn.Module):
    def __init__(self, in_features, out_features, k_neighbors=10, n_clusters=10, proj_dim=64, dropout=0.0):
        super(ST_HGNN_Layer, self).__init__()
        self.k = k_neighbors
        self.n_clusters = n_clusters
        self.proj_dim = proj_dim
        self.dropout = dropout

        # Step 1: Feature Projection
        self.W_proj = nn.Linear(in_features, proj_dim)

        # Step 3: Cluster Centers
        # C in R^{K x proj_dim}
        self.C = nn.Parameter(torch.empty(n_clusters, proj_dim))
        nn.init.xavier_uniform_(self.C)

        # Convolution weights
        self.Theta = nn.Linear(in_features, out_features)
        
        # Activation
        self.act = nn.ELU()
        
    def forward(self, x):
        # x: (B, N, M, D)
        B, N, M, D = x.shape
        
        # Process as (B*M, N, D)
        x_flat = x.permute(0, 2, 1, 3).reshape(B*M, N, D)
        
        # 1. Projection
        Z = self.W_proj(x_flat) # (BM, N, proj_dim)
        
        # 2. k-NN Hyperedge
        # Calculate distances
        dist = torch.cdist(Z, Z)
        # Get k nearest neighbors
        # Note: topk returns the smallest distances
        k = min(self.k, N) # Ensure k <= N
        _, indices = dist.topk(k, dim=-1, largest=False) # (BM, N, k)
        
        # Construct H_knn
        H_knn = torch.zeros(B*M, N, N, device=x.device)
        H_knn.scatter_(2, indices, 1.0) # H[b, i, j] = 1 if j in indices[i]
        
        # 3. Clustering Hyperedge
        # H_cluster columns are cluster centers.
        H_cluster = F.softmax(torch.matmul(Z, self.C.t()), dim=-1) # (BM, N, K)
        
        # 4. Concatenate
        # H: (BM, N, N+K)
        H = torch.cat([H_knn, H_cluster], dim=-1)
        
        # 5. Convolution
        output = self.hypergraph_conv(x_flat, H)
        
        # Reshape back
        output = output.reshape(B, M, N, -1).permute(0, 2, 1, 3)
        return output

    def hypergraph_conv(self, X, H):
        # X: (Batch, N, D)
        # H: (Batch, N, Edges)
        
        # D_e: (Batch, Edges)
        D_e = H.sum(dim=1)
        D_e_inv = D_e.pow(-1)
        D_e_inv[torch.isinf(D_e_inv)] = 0
        
        # D_v: (Batch, N)
        D_v = H.sum(dim=2)
        D_v_inv_sqrt = D_v.pow(-0.5)
        D_v_inv_sqrt[torch.isinf(D_v_inv_sqrt)] = 0
        
        # Theta transform
        X_trans = self.Theta(X) # (Batch, N, D_out)
        
        # Propagation: D_v^-0.5 H W_e D_e^-1 H.T D_v^-0.5 X_trans
        # Assuming W_e = I
        
        # Step 1: D_v^-0.5 * X_trans
        out = D_v_inv_sqrt.unsqueeze(-1) * X_trans
        
        # Step 2: H.T @ out
        out = torch.matmul(H.transpose(1, 2), out) # (Batch, Edges, D_out)
        
        # Step 3: D_e^-1 * out
        out = D_e_inv.unsqueeze(-1) * out
        
        # Step 4: H @ out
        out = torch.matmul(H, out) # (Batch, N, D_out)
        
        # Step 5: D_v^-0.5 * out
        out = D_v_inv_sqrt.unsqueeze(-1) * out
        
        return self.act(out)

import torch.nn as nn
import dgl
import torch
import torch.nn as nn
from utils.layers.hyp_layers import *
from utils.manifolds import Euclidean 


class AMRComEnc(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        max_comment_count,
        comment_curvature,
    ):
        super(AMRComEnc, self).__init__()
        self.manifold = Euclidean() 
        self.c = comment_curvature

        self.conv1 = HGCNLayer(
            self.manifold,
            in_dim,
            hidden_dim,
            c_in=self.c,
            c_out=self.c,
            act=torch.tanh,
            dropout=0.1,
            use_bias=True,
        )
        self.conv2 = HGCNLayer(
            self.manifold,
            hidden_dim,
            hidden_dim,
            c_in=self.c,
            c_out=self.c,
            act=torch.tanh,
            dropout=0.1,
            use_bias=True,
        )
        self.hidden_dim = hidden_dim
        self.max_comment_count = max_comment_count
        
    def forward(self, g, h, subgraphs):
        """returned shape will be [batch_size, max_comments, embedding_size] i.e.
        [batch_size, max_comment_count, hidden_dim]"""

        # Apply graph convolution and activation.
        adj = g.adj()  # finding the adjacency matrix
        inp = h  # convertng to sparse tensor

        out, adj = self.conv1((inp, adj))
        out, adj = self.conv2((out, adj))
        h = out  # converting back to dense
        if self.content_module:
            with g.local_scope():
                g.ndata["h"] = h
                # Calculate graph representation by average readout.
                unbatched = dgl.unbatch(g)
                batch_agg = []
                for batch_idx in range(len(unbatched)):
                    agg = []
                    for node_list in subgraphs[batch_idx]:
                        sub = dgl.node_subgraph(unbatched[batch_idx], node_list)
                        hg = dgl.mean_nodes(sub, "h")
                        agg.append(torch.squeeze(hg).unsqueeze(0))
                    if len(agg) >= self.max_comment_count:
                        agg = agg[: self.max_comment_count]
                        agg = torch.cat([i.float() for i in agg], dim=0)
                    else:
                        padding = torch.zeros(
                            (self.max_comment_count - len(agg), self.hidden_dim),
                            dtype=torch.float32,
                            requires_grad=True,
                        )
                        without_padding = torch.cat([i.float() for i in agg], dim=0)
                        agg = torch.cat([without_padding, padding], dim=0)
                    batch_agg.append(agg.unsqueeze(0))
                ret = torch.cat(batch_agg, dim=0)
                return ret

        else:
            with g.local_scope():
                g.ndata["h"] = h
                ret = dgl.mean_nodes(g, "h")
                return ret

import torch.nn as nn
import dgl
import torch
import torch.nn as nn
from ..utils.layers.layers import * 


class ComEnc(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        max_comment_count,
        device,
    ):
        super(ComEnc, self).__init__()
        self.conv1 = GCNConv(
            in_dim,
            hidden_dim,
            bias=True,
        )
        self.act1 = nn.Tanh()
        self.conv2 = GCNConv(
            hidden_dim,
            hidden_dim,
            bias=True,
        )
        self.act2 = nn.Tanh()
        self.max_comment_count = max_comment_count
        self.hidden_dim = hidden_dim
        self.device = device

    def forward(self, g, h, subgraphs):
        """returned shape will be [batch_size, max_comments, embedding_size] i.e.
        [batch_size, max_comment_count, hidden_dim]"""

        # Apply graph convolution and activation.
        adj = g.adj().to(self.device)  # finding the adjacency matrix
        inp = h.to(self.device)  # convertng to sparse tensor

        out, adj = self.conv1((inp, adj))
        out= self.act1(out)
        out, adj = self.conv2((out, adj))
        out = self.act2(out)
        # print(out)
        h = out  # converting back to dense
        h = self.manifold.logmap0(self.manifold.proj(h, c=self.c), c=self.c)
        # map h (which is in poincare space/euclidean) to tangential space to aggregate the node representations
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
                    ).to(self.device)
                    without_padding = torch.cat([i.float() for i in agg], dim=0)
                    agg = torch.cat([without_padding, padding], dim=0)
                agg = self.manifold.proj(
                    self.manifold.expmap0(agg, c=self.c), c=self.c
                )
                batch_agg.append(agg.unsqueeze(0))
            ret = torch.cat(batch_agg, dim=0)
            return ret

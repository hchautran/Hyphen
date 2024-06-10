import torch
import torch.nn as nn
from ..s4d import S4D
from utils.manifolds import Euclidean

if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d

class S4Model(nn.Module):

    def __init__(
        self,
        d_input=100,
        d_model=256,
        d_output=100,
        n_layers=1,
        d_state=64,
        dropout=0.1,
        prenorm=False,
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, d_state=d_state, dropout=dropout, transposed=True, lr=min(0.0001, 0.01))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.GELU(),
            nn.Linear(d_model*2, d_output),
        ) 


    def forward(self, x:torch.Tensor, pooling=False):
        """
        Input x is shape (B, L, d_input)
        """
        # print(x.shape)
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)
        

        # Pooling: average pooling over the sequence length
        if pooling:
            x = x.mean(dim=1)
        x = self.decoder(x)  #

        return x


class S4DEnc(nn.Module):
    def __init__(
        self, 
        word_hidden_size, 
        sent_hidden_size, 
        batch_size, 
        embedding_matrix, 
        max_sent_length, 
        max_word_length, 
        device, 
        content_curvature
    ):
        super(S4DEnc, self).__init__()
        self.batch_size = batch_size
       
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.manifold = Euclidean() 
        self.content_curvature = content_curvature

        self.word_ssm= S4Model(d_input=word_hidden_size, d_model=word_hidden_size//2, d_output=sent_hidden_size, n_layers=1, prenorm=False, d_state=128)
        self.sent_ssm= S4Model(d_input=sent_hidden_size, d_model=sent_hidden_size//2, d_output=sent_hidden_size, n_layers=1, prenorm=False, d_state=128)
        self.lookup = self.create_embeddeding_layer(embedding_matrix)

    def forward(self, input):
        output_list = []
        # input = input.permute(1, 0, 2)
        for x in input:
            x = self.lookup(x)
            output = self.word_ssm(x=x, pooling=True) 
            output_list.append(output)
        output = torch.stack(output_list, dim=0)
        x = self.sent_ssm(output)
        clip_r = 2.0
        # return output
        x_norm = torch.norm(x, dim=-1, keepdim=True) + 1e-5
        fac = torch.minimum(torch.ones_like(x_norm), clip_r/ x_norm)
        x = x * fac
        
        return  self.manifold.expmap0(x, c=1.0)

    def create_embeddeding_layer(self, weights_matrix, trainable=False):
        self.num_embeddings, self.embedding_dim = weights_matrix.shape
        weights_matrix = torch.from_numpy(weights_matrix)
        emb_layer = nn.Embedding(self.num_embeddings, self.embedding_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        emb_layer.weight.requires_grad = trainable
        return emb_layer


        

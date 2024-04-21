import torch
from hyptorch.geoopt import ManifoldParameter, ManifoldTensor 
from hyptorch.geoopt.manifolds.stereographic.math import mobius_add, mobius_matvec
from hyptorch.geoopt.manifolds.lorentz import math as lmath
from hyptorch.lorentz.manifold import CustomLorentz
from hyptorch.lorentz.layers.LAttn import CrossAttention 
from utils.manifolds import PoincareBall
from lcoattention import CoAttention 
from pcoattention import CoAttention 
from transformers import CLIPTextConfig 


def test():
    a = torch.ones((1, 30))
    b = torch.ones((1, 30))
    c = a + b
    return c

def lorentz_addition(x, y):
    x_p = lmath.lorentz_to_poincare(x, k=manifold.k)
    y_p = lmath.lorentz_to_poincare(y, k=manifold.k)
    return lmath.poincare_to_lorentz(mobius_add(x_p, y_p, k=manifold.k),k=manifold.k)

def direct_add(x, y):
    x = x.narrow(-1, 1, y.shape[-1] - 1) + y.narrow(
        -1, 1, y.shape[-1] - 1
    )
    hidden_states = manifold.add_time(hidden_states)


if __name__ == "__main__":
    manifold = CustomLorentz(k=1)
    # a = ManifoldParameter(manifold=manifold, data=manifold.random_normal((4, 4)))
    # b = ManifoldParameter(manifold=manifold, data=manifold.random_normal((4,4)))
    # x = ManifoldTensor(manifold.random_normal , manifold=manifold)
    # theta = torch.nn.Parameter(torch.tensor(0.3)) 
    # gamma = torch.nn.Parameter(torch.tensor(0.5)) 
    x = manifold.random_normal(32,20, 101)
    y = manifold.random_normal(32,10, 101)

    manifold.assert_check_point_on_manifold(x)
    
    manifold.assert_check_point_on_manifold(y)

    x_p = lmath.lorentz_to_poincare(x, k=manifold.k)
    y_p = lmath.lorentz_to_poincare(y, k=manifold.k)
    # z = lmath.poincare_to_lorentz(mobius_add(x_p, y_p, k=manifold.k),k=manifold.k)
    # manifold.assert_check_point_on_manifold(z)


    x = manifold.random_normal(32, 20, 101)
    y = manifold.random_normal(32, 10, 101)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # model = CrossAttention(manifold=manifold, embedding_dim=100 )
    # sc, a_s, a_c= model(x, y)
    # print(a_s.shape)
    # print(a_c.shape)
    # co_model = CoAttention(manifold=PoincareBall(), device=device)
    # sc, a_s, a_c= co_model(x_p, y_p)
    # print(a_s.shape)
    # print(a_c.shape)

    config = CLIPTextConfig(
       hidden_size=100, 
       num_attention_heads=2
    )
    
    model = CrossAttention(manifold, config)
    x, attn = model(y, x, output_attentions=True) 
    y, attn = model(x, y, output_attentions=True) 

    print(x.shape)
    print(y.shape)



        



import torch 


eps = 1e-7
def E2Lorentz(input):
    """Function to convert fromm Euclidean space to the Lorentz model"""
    rr = torch.norm(input, p=2, dim=2)
    dd = input.permute(2,0,1) / rr
    cosh_r = torch.cosh(rr)
    sinh_r = torch.sinh(rr)
    output = torch.cat(((dd * sinh_r).permute(1, 2, 0), cosh_r.unsqueeze(0).permute(1, 2, 0)), dim=2)
    return output

def P2Lorentz(input):
    """Function to convert fromm Poincare model to the Lorentz model"""
    rr = torch.norm(input, p=2, dim=2)
    output = torch.cat((2*input, (1+rr**2).unsqueeze(2)),dim=2).permute(2,0,1)/(1-rr**2+eps)
    return output.permute(1,2,0)

def L2Klein(input):
    """Function to convert fromm Lorentz model to the Klein model"""
    dump = input[:, :, -1]
    dump = torch.clamp(dump, eps, 1.0e+16)
    return (input[:, :, :-1].permute(2, 0, 1)/dump).permute(1, 2, 0)

def arcosh(x):
    c0 = torch.log(x)
    c1 = torch.log1p(torch.sqrt(x * x - 1 + eps) / x)
    return c0 + c1

def disLorentz(x, y):
    m = x * y
    prod_minus = -m[:, :, :-1].sum(dim=2) + m[:, :, -1]
    output = torch.clamp(prod_minus, 1.0 + eps, 1.0e+16)
    return arcosh(output)
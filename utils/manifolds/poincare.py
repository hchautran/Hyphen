import torch
from utils.manifolds.base import Manifold
from utils.utils import artanh, tanh

class PoincareBall(Manifold):
    """
    PoicareBall Manifold class.

    We use the following convention: x0^2 + x1^2 + ... + xd^2 < 1 / c

    Note that 1/sqrt(c) is the Poincare ball radius.

    """

    def __init__(self, c):
        super(PoincareBall, self).__init__()
        self.name = 'PoincareBall'
        self.min_norm = 1e-14
        self.c= c 
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}

    def norm(self, p):
        p_norm = torch.sqrt(p.pow(2).sum(-1) + 1e-10).unsqueeze(-1)
        return p_norm

    def sqdist(self, p1, p2):
        sqrt_c = self.c ** 0.5
        test = sqrt_c * self.mobius_add(self.proj(-p1, self.c), self.proj(p2, self.c), self.c, dim=-1).norm(dim=-1, p=2, keepdim=False)
        assert torch.max(test) <= 1
        assert torch.min(test) >= -1
        dist_c = artanh(sqrt_c * self.mobius_add(-p1, p2, self.c, dim=-1).norm(dim=-1, p=2, keepdim=False))
        dist = dist_c * 2 / sqrt_c
        return dist ** 2

    def _lambda_x(self, x):
        x_sqnorm = torch.sum(x.data.pow(2), dim=-1, keepdim=True)
        return 2 / (1. - self.c * x_sqnorm).clamp_min(self.min_norm)

    def egrad2rgrad(self, p, dp):
        lambda_p = self._lambda_x(p, self.c)
        dp /= lambda_p.pow(2)
        return dp

    def proj(self, x):
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / (self.c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def proj_tan(self, u, p):
        return u

    def proj_tan0(self, u ):
        return u

    def expmap(self, u, p):
        sqrt_c = self.c ** 0.5
        u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        second_term = (
                tanh(sqrt_c / 2 * self._lambda_x(p, self.c) * u_norm)
                * u
                / (sqrt_c * u_norm)
        )
        gamma_1 = self.mobius_add(p, second_term, self.c)
        return gamma_1

    def logmap(self, p1, p2):
        sub = self.mobius_add(-p1, p2, self.c)
        sub_norm = sub.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        lam = self._lambda_x(p1, self.c)
        sqrt_c = self.c ** 0.5
        return 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm

    def expmap0(self, u):
        sqrt_c = self.c ** 0.5
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), self.min_norm)
        gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma_1

    def logmap0(self, p, monitor = False):
        sqrt_c = self.c ** 0.5
        p_norm = p.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        if monitor: print(p_norm)
        scale = 1. / sqrt_c * artanh(sqrt_c * p_norm) / p_norm
        return scale * p

    def mobius_add(self, x, y, dim=-1):
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        xy = (x * y).sum(dim=dim, keepdim=True)
        num = (1 + 2 * self.c * xy + self.c * y2) * x + (1 - self.c * x2) * y
        denom = 1 + 2 * self.c * xy + self.c ** 2 * x2 * y2
        return num / denom.clamp_min(self.min_norm)

    def mobius_matvec(self, m, x,  monitor = False):
        x = self.proj(x, self.c)
        u = self.logmap0(x, self.c)
        mu = u @ m.transpose(-1, -2)
        if monitor: print(x, u, mu, self.expmap0(mu, self.c))
        return self.expmap0(mu, self.c)

    def init_weights(self, w, irange=1e-5):
        w.data.uniform_(-irange, irange)
        return w

    def _gyration(self, u, v, w, dim: int = -1):
        u2 = u.pow(2).sum(dim=dim, keepdim=True)
        v2 = v.pow(2).sum(dim=dim, keepdim=True)
        uv = (u * v).sum(dim=dim, keepdim=True)
        uw = (u * w).sum(dim=dim, keepdim=True)
        vw = (v * w).sum(dim=dim, keepdim=True)
        c2 = self.c ** 2
        a = -c2 * uw * v2 + self.c * vw + 2 * c2 * uv * vw
        b = -c2 * vw * u2 - self.c * uw
        d = 1 + 2 * self.c * uv + c2 * u2 * v2
        return w + 2 * (a * u + b * v) / d.clamp_min(self.min_norm)

    def inner(self, x, u, v=None, keepdim=False, dim=-1):
        if v is None:
            v = u
        lambda_x = self._lambda_x(x, self.c)
        return lambda_x ** 2 * (u * v).sum(dim=dim, keepdim=keepdim)

    def ptransp(self, x, y, u):
        lambda_x = self._lambda_x(x, self.c)
        lambda_y = self._lambda_x(y, self.c)
        return self._gyration(y, -x, u, self.c) * lambda_x / lambda_y

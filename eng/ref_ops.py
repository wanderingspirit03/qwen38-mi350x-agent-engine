"""Plain PyTorch versions of the model's operations: the definitions the fast kernels are checked against."""
import torch
import torch.nn.functional as F


def rmsnorm(x, w, eps=1e-6):
    """Qwen3.5 norm: x / rms(x) * (1 + w), computed in fp32."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * (1.0 + w.float())).to(x.dtype)


def rmsnorm_gated(x, z, w, eps=1e-6):
    """GDN output norm: (x / rms(x) * w) * silu(z); w plain (not 1 + w)."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    xf = (w * xf.to(x.dtype)).float()
    return (xf * F.silu(z.float())).to(x.dtype)


def l2norm(x, eps=1e-6):
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).sum(-1, keepdim=True) + eps)


def gdn_recurrent(q, k, v, g, beta, S):
    """q, k: (T, H, dk) fp32 (q already scaled), v: (T, H, dv), g, beta: (T, H); S: (H, dk, dv) fp32, updated
    in place. Returns o (T, H, dv) fp32."""
    T = q.shape[0]
    out = torch.empty(v.shape, dtype=torch.float32, device=v.device)
    for t in range(T):
        S.mul_(g[t].exp()[:, None, None])
        kv = torch.einsum("hk,hkv->hv", k[t], S)
        delta = (v[t].float() - kv) * beta[t][:, None]
        S.add_(k[t][:, :, None] * delta[:, None, :])
        out[t] = torch.einsum("hk,hkv->hv", q[t], S)
    return out


def gdn_chunked(q, k, v, g, beta, S, C=64):
    """Same as gdn_recurrent, by chunks of C positions (the WY form). S updated in place."""
    T, H, dk = q.shape
    dv = v.shape[-1]
    pad = (-T) % C
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, 0, 0, pad)) for x in (q, k, v))
        g, beta = (F.pad(x, (0, 0, 0, pad)) for x in (g, beta))
    n = (T + pad) // C
    # (H, n, C, d)
    q, k, v = (x.float().reshape(n, C, H, -1).permute(2, 0, 1, 3) for x in (q, k, v))
    g, beta = (x.float().reshape(n, C, H).permute(2, 0, 1) for x in (g, beta))
    G = g.cumsum(-1)                                            # (H, n, C)
    decay = (G[..., :, None] - G[..., None, :])                 # i, j: G_i - G_j
    lower = torch.ones(C, C, dtype=torch.bool, device=q.device).tril(-1)
    lower_eq = torch.ones(C, C, dtype=torch.bool, device=q.device).tril(0)
    D = torch.where(lower, decay, torch.full_like(decay, float("-inf"))).exp()
    Deq = torch.where(lower_eq, decay, torch.full_like(decay, float("-inf"))).exp()
    kb = k * beta[..., None]
    A = (kb @ k.transpose(-1, -2)) * D                          # strictly lower
    I = torch.eye(C, device=q.device).expand_as(A)
    Tm = torch.linalg.solve_triangular(I + A, I, upper=False)   # (I + A)^-1
    w = Tm @ (kb * G.exp()[..., None])
    u = Tm @ (v * beta[..., None])
    att = (q @ k.transpose(-1, -2)) * Deq
    o = torch.empty(H, n, C, dv, device=q.device)
    for c in range(n):
        vn = u[:, c] - w[:, c] @ S
        o[:, c] = (q[:, c] * G[:, c].exp()[..., None]) @ S + att[:, c] @ vn
        gl = G[:, c, -1]
        S.mul_(gl.exp()[:, None, None])
        S.add_((k[:, c] * (gl[:, None] - G[:, c]).exp()[..., None]).transpose(-1, -2) @ vn)
    return o.permute(1, 2, 0, 3).reshape(n * C, H, dv)[:T]

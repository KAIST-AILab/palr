import torch

def compute_pdist_sq(x, y=None):
    """compute the squared paired distance between x and y."""
    if y is not None:
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        return torch.clamp(x_norm + y_norm - 2.0 * x @ y.T, min=0)
    a = x.view(x.shape[0], -1)
    aTa = torch.mm(a, a.T)
    aTa_diag = torch.diag(aTa)
    aTa = torch.clamp(aTa_diag + aTa_diag.unsqueeze(-1) - 2 * aTa, min=0)

    ind = torch.triu_indices(x.shape[0], x.shape[0], offset=1, device=x.device)
    aTa[ind[0], ind[1]] = 0
    return aTa + aTa.transpose(0, 1)

def gaussian_kernel(X, sigma2=None, Y=None, normalized=False, **ignored):
    if normalized:
        X = X / torch.linalg.norm(X, dim=1, keepdim=True)
        if Y is not None:
            Y = Y / torch.linalg.norm(Y, dim=1, keepdim=True)
    Dxx = compute_pdist_sq(X, Y)
    if sigma2 is None:
        sigma2 = Dxx.median()
    Kx = torch.exp(-Dxx / sigma2)
    return Kx

def estimate_hscic(X, Y, Z, ridge_lambda=1e-2, use_median=False, normalize_kernel=False, sigma2=None):
    '''X ind. Y | Z '''
    # https://arxiv.org/pdf/2207.09768.pdf
    # (1) action regularization version : X = imitator action
    # (2) regularized representation version : X = varphi(Obs)    

    if sigma2 is None:
        if use_median:
            sigma2_ = None
        else:
            sigma2_ = 1.
    else:
        sigma2_ = sigma2

    Kx = gaussian_kernel(X, sigma2=sigma2_, normalized=normalize_kernel)
    Ky = gaussian_kernel(Y, sigma2=sigma2_, normalized=normalize_kernel)
    Kz = gaussian_kernel(Z, sigma2=sigma2_, normalized=normalize_kernel)
    
    n = Kz.shape[0]    
    WtKzz = torch.linalg.solve(Kz + ridge_lambda * n * torch.eye(n).to(Kz.device), Kz) 
    # * Kz.shape[0] for ridge_lambda
    
    term_1 = (WtKzz * ((Kx * Ky) @ WtKzz)).sum()    # tr(WtKzz.T @ (Kx * Ky) @ WtKzz)
    WkKxWk = WtKzz * (Kx @ WtKzz)
    KyWk = Ky @ WtKzz
    term_2 = (WkKxWk * KyWk).sum()        
    term_3 = (WkKxWk.sum(dim=0) * (WtKzz * KyWk).sum(dim=0)).sum()
    
    # W = (Kz + ridge_lambda  * torch.eye(Kz.shape[0])).inverse()
    # term1 = Kz.T @ W @ (Kx * Ky) @ W.T @ Kz
    # term2 = -2 * Kz.T @ W ( Kx@W.T@Kz * Ky@W.T@Kz )
    # term3 = (Kz.T @ W @ Kx @ W.T @ Kz) * (Kz.T @ W @ Ky @ W.T @ Kz)

    return (term_1 - 2 * term_2 + term_3) / Kz.shape[0]
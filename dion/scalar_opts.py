import torch
from torch import Tensor
from collections.abc import Generator


@torch.compile(fullgraph=True)
def adamw_update(
    X: Tensor,  # Model weights (modified in place)
    G: Tensor,  # Gradient
    M: Tensor,  # Momentum buffer (modified in place)
    V: Tensor,  # Variance buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    step: int,
    epsilon: float,
    cautious_wd: bool = False,
):
    """
    AdamW optimizer algorithm.
    """
    assert X.shape == G.shape
    assert X.shape == M.shape

    # Update momentum and variance
    # M = beta1 * M + (1 - beta1) * G
    M.lerp_(G.to(M.dtype), 1 - beta1)
    # V = beta2 * V + (1 - beta2) * G * G
    V.mul_(beta2).addcmul_(G, G, value=1 - beta2)  # type: ignore[arg-type]

    # Bias correction
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    bias_correction2_sqrt = bias_correction2.sqrt()

    # The goal is to compute the following in-place:
    # M = M / bias_correction1
    # V = V / bias_correction2
    # X = X - lr * M / (sqrt(V) + epsilon)

    # sqrt(V / bias_correction2) = sqrt(V) / sqrt(bias_correction2)
    denom = V.sqrt().div_(bias_correction2_sqrt).add_(epsilon)

    # Adjust learning rate to include bias correction 1
    adj_lr = lr / bias_correction1

    if cautious_wd:
        # Compute update direction (pre-LR) for CWD mask
        update_dir = M / denom

        # Apply cautious weight decay: only where update and parameter signs align
        # Reference: https://arxiv.org/pdf/2510.12402
        coeff = lr * weight_decay
        decay_mask = (update_dir * X >= 0).to(dtype=X.dtype)
        decay = (X * decay_mask) * coeff
        X.sub_(decay)
    else:
        # Apply weight decay
        X.mul_(1 - lr * weight_decay)

    # Weight update
    # X = X - adj_lr * M / denom
    X.addcdiv_(M, denom, value=-adj_lr)  # type: ignore[arg-type]


@torch.compile(fullgraph=True)
def lion_update(
    X: Tensor,  # Model weights (modified in place)
    G: Tensor,  # Gradient
    M: Tensor,  # Momentum buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    cautious_wd: bool = False,
):
    """
    Lion optimizer algorithm. Sign update should guarantee RMS norm equal to 1.
    """
    assert X.shape == G.shape
    assert X.shape == M.shape

    G = G.to(M.dtype)

    # Compute sign update
    # U = sign(beta1 * M + (1 - beta1) * G)
    U = M.lerp(G, 1 - beta1).sign_()

    # Update momentum with new gradient
    # M = beta2 * M + (1 - beta2) * G
    M.lerp_(G, 1 - beta2)

    if cautious_wd:
        # Apply cautious weight decay: only where update and parameter signs align
        # Reference: https://arxiv.org/pdf/2510.12402
        coeff = lr * weight_decay
        decay_mask = (U * X >= 0).to(dtype=X.dtype)
        decay = (X * decay_mask) * coeff
        X.sub_(decay)
    else:
        # Apply weight decay
        X.mul_(1 - lr * weight_decay)

    # Weight update
    # X = X - lr * U
    X.add_(U, alpha=-lr)  # type: ignore[arg-type]


@torch.compile(fullgraph=True)
def adamw_update_foreach(
    X: list[Tensor],  # Model weights (modified in place)
    G: list[Tensor],  # Gradient
    M: list[Tensor],  # Momentum buffer (modified in place)
    V: list[Tensor],  # Variance buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    step: Tensor,
    epsilon: Tensor,
    cautious_wd: bool = False,
    mask_scales: Tensor | None = None,
):
    """
    AdamW optimizer algorithm (foreach implementation).
    """
    batch_size = len(X)
    assert batch_size == len(G)
    assert batch_size == len(M)
    assert batch_size == len(V)
    if mask_scales is None:
        mask_scales = torch.ones(batch_size, device=X[0].device, dtype=torch.float32)
    assert mask_scales.ndim == 1
    assert batch_size == int(mask_scales.numel())
    per_param_scales = tuple(mask_scales.unbind())

    M_dtype = M[0].dtype
    V_dtype = V[0].dtype

    # Update momentum and variance
    # M = beta1 * M + (1 - beta1) * G
    G = [g.to(dtype=M_dtype) for g in G]
    torch._foreach_lerp_(M, G, [1 - beta1] * batch_size)

    # V = beta2 * V + (1 - beta2) * G * G
    G_square = torch._foreach_mul(G, G)
    G_square = [g.to(dtype=V_dtype) for g in G_square]
    torch._foreach_lerp_(V, G_square, [1 - beta2] * batch_size)

    # Bias correction
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    bias_correction2_sqrt = bias_correction2.sqrt()

    # The goal is to compute the following in-place:
    # M = M / bias_correction1
    # V = V / bias_correction2
    # X = X - lr * M / (sqrt(V) + epsilon)

    # Compute the denominator for the weight update
    # sqrt(V / bias_correction2) = sqrt(V) / sqrt(bias_correction2)
    denom = torch._foreach_sqrt(V)
    torch._foreach_div_(denom, bias_correction2_sqrt)
    torch._foreach_add_(denom, [epsilon] * batch_size)

    # Adjust learning rate to include bias correction 1
    adj_lr = lr / bias_correction1

    M_div = torch._foreach_div(M, denom)

    if cautious_wd:
        # Apply cautious weight decay: only where update and parameter signs align
        # Reference: https://arxiv.org/pdf/2510.12402
        coeff = lr * weight_decay

        decay_masks = torch._foreach_mul(X, M_div)
        decay_masks = torch._foreach_sign(decay_masks)  # {-1, 0, 1}
        decay_masks = torch._foreach_add(decay_masks, 1)  # {0, 1, 2}
        decay_masks = torch._foreach_minimum(decay_masks, 1)  # {0, 1, 1}

        decay_terms = torch._foreach_mul(X, decay_masks)
        torch._foreach_mul_(decay_terms, coeff)
        torch._foreach_sub_(X, decay_terms)
    else:
        # Apply weight decay
        torch._foreach_mul_(X, 1 - lr * weight_decay)

    # Weight update
    # X = X - adj_lr * M / denom
    torch._foreach_mul_(M_div, adj_lr)
    torch._foreach_mul_(M_div, per_param_scales)
    torch._foreach_sub_(X, M_div)


@torch.compile(fullgraph=True)
def lion_update_foreach(
    X: list[Tensor],  # Model weights (modified in place)
    G: list[Tensor],  # Gradient
    M: list[Tensor],  # Momentum buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    cautious_wd: bool = False,
    mask_scales: Tensor | None = None,
):
    """
    Lion optimizer algorithm (foreach implementation).
    """
    batch_size = len(X)
    assert batch_size == len(G)
    assert batch_size == len(M)
    if mask_scales is None:
        mask_scales = torch.ones(batch_size, device=X[0].device, dtype=torch.float32)
    assert mask_scales.ndim == 1
    assert batch_size == int(mask_scales.numel())
    per_param_scales = tuple(mask_scales.unbind())

    # 不能用 `M[0].dtype` 作为全局 dtype：参数/动量 buffer 可能混合 float32/float64；
    # foreach API 允许列表中不同 dtype 的 tensor，但要求每对 (start, end) dtype 一致。
    # G = [g.to(dtype=m.dtype) for g, m in zip(G, M, strict=True)]
    dtype = M[0].dtype
    G = [g.to(dtype=dtype) for g in G]

    # Compute sign update
    # U = sign(beta1 * M + (1 - beta1) * G)
    U = torch._foreach_lerp(M, G, [1 - beta1] * batch_size)
    torch._foreach_sign_(U)

    # Update momentum in place with new gradient
    # M = beta2 * M + (1 - beta2) * G
    torch._foreach_lerp_(M, G, [1 - beta2] * batch_size)

    if cautious_wd:
        # Apply cautious weight decay: only where update and parameter signs align
        # Reference: https://arxiv.org/pdf/2510.12402
        coeff = lr * weight_decay

        decay_masks = torch._foreach_mul(X, U)
        decay_masks = torch._foreach_sign(decay_masks)  # {-1, 0, 1}
        decay_masks = torch._foreach_add(decay_masks, 1)  # {0, 1, 2}
        decay_masks = torch._foreach_minimum(decay_masks, 1)  # {0, 1, 1}

        decay_terms = torch._foreach_mul(X, decay_masks)
        torch._foreach_mul_(decay_terms, coeff)
        torch._foreach_sub_(X, decay_terms)
    else:
        # Apply weight decay
        torch._foreach_mul_(X, 1 - lr * weight_decay)

    # Weight update
    # X = X - lr * U
    torch._foreach_mul_(U, lr)
    torch._foreach_mul_(U, per_param_scales)
    torch._foreach_sub_(X, U)


def adamw_update_foreach_async(
    X: list[Tensor],
    G: list[Tensor],
    M: list[Tensor],
    V: list[Tensor],
    lr: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    weight_decay: Tensor,
    step: Tensor,
    epsilon: Tensor,
    cautious_wd: bool = False,
    mask_scales: Tensor | None = None,
) -> Generator[None, None, None]:
    if mask_scales is None:
        mask_scales = torch.ones(len(X), device=X[0].device, dtype=torch.float32)
    adamw_update_foreach(X, G, M, V, lr, beta1, beta2, weight_decay, step, epsilon, cautious_wd, mask_scales)
    yield


def lion_update_foreach_async(
    X: list[Tensor],
    G: list[Tensor],
    M: list[Tensor],
    lr: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    weight_decay: Tensor,
    cautious_wd: bool = False,
    mask_scales: Tensor | None = None,
) -> Generator[None, None, None]:
    if mask_scales is None:
        mask_scales = torch.ones(len(X), device=X[0].device, dtype=torch.float32)
    lion_update_foreach(X, G, M, lr, beta1, beta2, weight_decay, cautious_wd, mask_scales)
    yield

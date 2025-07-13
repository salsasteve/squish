import torch


def get_svd_factors(
    weight: torch.Tensor, on_cpu: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Performs SVD on a weight tensor, ensuring results are on the original device."""
    original_device = weight.device

    # Determine the device for SVD computation
    # If on_cpu is True, move to CPU, otherwise use the tensor's current device
    weight_for_svd = weight.cpu() if on_cpu else weight

    # Perform SVD on the chosen device
    U, S, Vt = torch.linalg.svd(weight_for_svd.float(), full_matrices=False)

    # Always ensure the output tensors are on the same device as the input tensor
    if U.device != original_device:
        U = U.to(original_device)
        S = S.to(original_device)
        Vt = Vt.to(original_device)

    return U, S, Vt


def calculate_rank_from_svd(S: torch.Tensor, target_retention: float) -> int:
    """Calculates the rank needed to retain a target amount of energy."""
    energy = S.pow(2)
    total_energy = energy.sum()
    cumulative_energy = torch.cumsum(energy, dim=0)
    energy_ratios = cumulative_energy / total_energy

    target_idx = torch.where(energy_ratios >= target_retention)[0]
    optimal_rank = target_idx[0].item() + 1 if len(target_idx) > 0 else len(S)

    return int(max(4, min(optimal_rank, len(S) - 1)))


def create_low_rank_factors(
    U: torch.Tensor, S: torch.Tensor, Vt: torch.Tensor, rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Creates the two low-rank matrices U and V from SVD results."""
    U_truncated = U[:, :rank]
    S_sqrt_diag = torch.diag(torch.sqrt(S[:rank]))
    Vt_truncated = Vt[:rank, :]

    # Absorb the singular values into U and V
    U_factor = U_truncated @ S_sqrt_diag
    V_factor = S_sqrt_diag @ Vt_truncated

    return U_factor, V_factor

import torch
import pytest
from squish.svd import get_svd_factors, calculate_rank_from_svd, create_low_rank_factors


@pytest.fixture
def sample_tensor():
    """Provides a sample tensor for testing."""
    return torch.randn(10, 5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.fixture
def sample_tensor_gpu():
    """Provides a sample tensor on the GPU."""
    return torch.randn(10, 5).cuda()


def test_svd_factors_on_cpu(sample_tensor):
    """Tests the get_svd_factors function with on_cpu=True."""
    U, S, Vt = get_svd_factors(sample_tensor, on_cpu=True)

    # Check if the outputs are torch tensors
    assert isinstance(U, torch.Tensor)
    assert isinstance(S, torch.Tensor)
    assert isinstance(Vt, torch.Tensor)

    # Check the shapes of the output tensors
    assert U.shape == (10, 5)
    assert S.shape == (5,)
    assert Vt.shape == (5, 5)

    # Check if the reconstructed matrix is close to the original
    reconstructed_weight = U @ torch.diag(S) @ Vt
    assert torch.allclose(reconstructed_weight, sample_tensor, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_svd_factors_on_gpu():
    """Tests the get_svd_factors function on a GPU."""
    tensor_gpu = torch.randn(10, 5).cuda()
    U, S, Vt = get_svd_factors(tensor_gpu, on_cpu=False)

    # Check if the outputs are on the correct device
    assert U.device.type == "cuda"
    assert S.device.type == "cuda"
    assert Vt.device.type == "cuda"

    # Check the shapes of the output tensors
    assert U.shape == (10, 5)
    assert S.shape == (5,)
    assert Vt.shape == (5, 5)

    # Check if the reconstructed matrix is close to the original
    reconstructed_weight = U @ torch.diag(S) @ Vt
    assert torch.allclose(reconstructed_weight, tensor_gpu, atol=1e-5)


def test_return_to_original_device(sample_tensor):
    """Tests that the tensors are returned to the original device."""
    device = sample_tensor.device
    U, S, Vt = get_svd_factors(sample_tensor, on_cpu=True)

    assert U.device == device
    assert S.device == device
    assert Vt.device == device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_svd_factors_on_cpu_from_gpu_tensor(sample_tensor_gpu):
    """
    Tests SVD on a GPU tensor with computation forced to the CPU.
    This test specifically covers the lines that move the results back to the GPU.
    """
    # The original tensor is on the GPU
    original_device = sample_tensor_gpu.device
    assert original_device.type == "cuda"

    # We run the SVD on the CPU
    U, S, Vt = get_svd_factors(sample_tensor_gpu, on_cpu=True)

    # Assert that the results have been moved BACK to the original GPU device
    assert U.device == original_device, "U was not moved back to the original device"
    assert S.device == original_device, "S was not moved back to the original device"
    assert Vt.device == original_device, "Vt was not moved back to the original device"

    # Verify the reconstruction is still correct
    reconstructed = U @ torch.diag(S) @ Vt
    assert torch.allclose(reconstructed, sample_tensor_gpu, atol=1e-5)


def test_basic_rank_calculation():
    """
    Tests the basic functionality of the rank calculation.
    """
    S = torch.tensor([10.0, 5.0, 2.0, 1.0, 0.5, 0.1])
    target_retention = 0.95
    # Expected energy: [100, 25, 4, 1, 0.25, 0.01]
    # Total energy: 130.26
    # Cumulative energy: [100, 125, 129, 130, 130.25, 130.26]
    # Ratios: [0.767, 0.96, 0.99, 0.998, 0.999, 1.0]
    # The first index to meet 0.95 is 1 (0-indexed), so rank is 1 + 1 = 2.
    # The function then takes max(4, min(2, 5)) = 4
    assert calculate_rank_from_svd(S, target_retention) == 4


def test_full_retention():
    """
    Tests the case where the target retention is 1.0.
    """
    S = torch.tensor([10.0, 1.0, 0.1])
    target_retention = 1.0
    # The rank should be len(S) -1 = 2, and since min(2,2) = 2,
    # the function should return max(4, 2) which is 4.
    assert calculate_rank_from_svd(S, target_retention) == 4


def test_low_retention():
    """
    Tests a low retention target to ensure the minimum rank is respected.
    """
    S = torch.tensor([100.0, 1.0, 0.5])
    target_retention = 0.1
    # The calculated rank will be 1, but the function enforces a minimum of 4.
    assert calculate_rank_from_svd(S, target_retention) == 4


def test_no_retention_met():
    """
    Tests the scenario where the target retention is never met.
    """
    S = torch.tensor([1.0, 1.0, 1.0])
    target_retention = 1.1  # Impossible to achieve
    # Should return len(S) - 1, which is 2. The function returns max(4, min(2,2)) = 4
    assert calculate_rank_from_svd(S, target_retention) == 4


def test_rank_upper_bound():
    """
    Ensures the rank does not exceed the maximum possible rank.
    """
    S = torch.tensor([1.0] * 20)
    target_retention = 0.99
    # The calculated rank should not be more than len(S) - 1 = 19.
    assert calculate_rank_from_svd(S, target_retention) <= 19


@pytest.fixture
def svd_components():
    """Provides a set of SVD components for testing."""
    original_matrix = torch.randn(20, 10)
    U, S, Vt = torch.linalg.svd(original_matrix, full_matrices=False)
    return U, S, Vt


def test_factor_shapes(svd_components):
    """Tests if the output factors have the correct shapes."""
    U, S, Vt = svd_components
    rank = 5

    U_factor, V_factor = create_low_rank_factors(U, S, Vt, rank)

    # Check the shapes of the output factors
    assert U_factor.shape == (20, rank), "Shape of U_factor is incorrect"
    assert V_factor.shape == (rank, 10), "Shape of V_factor is incorrect"


def test_reconstruction_correctness(svd_components):
    """
    Tests if the product of the low-rank factors correctly approximates
    the original matrix.
    """
    U, S, Vt = svd_components
    rank = 8

    U_factor, V_factor = create_low_rank_factors(U, S, Vt, rank)

    # Reconstruct the matrix from the low-rank factors
    reconstructed_matrix = U_factor @ V_factor

    # Create the true low-rank approximation for comparison
    true_low_rank_matrix = U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :]

    # Check that the reconstructed matrix is close to the true low-rank version
    assert torch.allclose(reconstructed_matrix, true_low_rank_matrix, atol=1e-6)


def test_full_rank_scenario(svd_components):
    """Tests the function when the rank is the full rank of the matrix."""
    U, S, Vt = svd_components
    full_rank = S.shape[0]  # This is 10 in this case

    U_factor, V_factor = create_low_rank_factors(U, S, Vt, full_rank)

    # Reconstruct the matrix from the factors
    reconstructed_matrix = U_factor @ V_factor

    # Reconstruct the original matrix from the full SVD components
    original_matrix_reconstructed = U @ torch.diag(S) @ Vt

    # Check if the reconstruction is very close to the original
    assert torch.allclose(
        reconstructed_matrix, original_matrix_reconstructed, atol=1e-6
    )

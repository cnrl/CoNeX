import torch


def DoGFilter(
    size,
    sigma_1,
    sigma_2,
    step=None,
    zero_mean=False,
    one_sum=False,
    device=None,
    dtype=None,
):
    step = 1 if step is None else step
    scale = (size - 1) / 2

    v_range = torch.arange(-scale, scale + step, step, dtype=dtype, device=device)
    x, y = torch.meshgrid(v_range, v_range, indexing="ij")

    g_values = -(x**2 + y**2) / 2

    dog_1 = torch.exp(g_values / (sigma_1**2)) / sigma_1
    dog_2 = torch.exp(g_values / (sigma_2**2)) / sigma_2

    dog_filter = (dog_1 - dog_2) / torch.sqrt(
        torch.tensor(2 * torch.pi, device=device, dtype=dtype)
    )

    if zero_mean:
        p_sum = torch.sum(dog_filter[dog_filter > 0])
        n_sum = torch.sum(dog_filter[dog_filter < 0])
        dog_filter[dog_filter < 0] *= -p_sum / n_sum

    if one_sum:
        dog_filter /= torch.sum(torch.abs(dog_filter))

    return dog_filter


def GaborFilter(
    size,
    labda,
    theta,
    sigma,
    gamma,
    step=None,
    zero_mean=False,
    one_sum=False,
    device=None,
    dtype=None,
):
    step = 1 if step is None else step
    scale = (size - 1) / 2

    v_range = torch.arange(-scale, scale + step, step, dtype=dtype, device=device)
    x, y = torch.meshgrid(v_range, v_range, indexing="ij")

    x_rotated = x * torch.cos(
        torch.tensor(theta, device=device, dtype=dtype)
    ) + y * torch.sin(torch.tensor(theta, device=device, dtype=dtype))
    y_rotated = -x * torch.sin(
        torch.tensor(theta, device=device, dtype=dtype)
    ) + y * torch.cos(torch.tensor(theta, device=device, dtype=dtype))

    gabor_filter = torch.exp(
        -(x_rotated**2 + (gamma**2 * y_rotated**2)) / (2 * sigma**2)
    ) * torch.cos(2 * torch.pi * x_rotated / labda)

    if zero_mean:
        p_sum = torch.sum(gabor_filter[gabor_filter > 0])
        n_sum = torch.sum(gabor_filter[gabor_filter < 0])
        gabor_filter[gabor_filter < 0] *= -p_sum / n_sum

    if one_sum:
        gabor_filter /= torch.sum(torch.abs(gabor_filter))

    return gabor_filter

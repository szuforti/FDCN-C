from numbers import Real

import numpy as np
from scipy.interpolate import Rbf
from sklearn.utils import check_random_state
import torch
from torch.fft import fft, ifft
from torch.nn.functional import pad, one_hot


def _new_random_fft_phase_odd(batch_size, c, n, device, random_state):
    rng = check_random_state(random_state)
    random_phase = torch.from_numpy(
        2j * np.pi * rng.random((batch_size, c, (n - 1) // 2))
    ).to(device)
    return torch.cat([
        torch.zeros((batch_size, c, 1), device=device),
        random_phase,
        -torch.flip(random_phase, [-1])
    ], dim=-1)


def _new_random_fft_phase_even(batch_size, c, n, device, random_state):
    rng = check_random_state(random_state)
    random_phase = torch.from_numpy(
        2j * np.pi * rng.random((batch_size, c, n // 2 - 1))
    ).to(device)
    return torch.cat([
        torch.zeros((batch_size, c, 1), device=device),
        random_phase,
        torch.zeros((batch_size, c, 1), device=device),
        -torch.flip(random_phase, [-1])
    ], dim=-1)


_new_random_fft_phase = {
    0: _new_random_fft_phase_even,
    1: _new_random_fft_phase_odd
}


def ft_surrogate(
        X,
        y,
        phase_noise_magnitude,
        channel_indep,
        random_state=None
):
    """FT surrogate augmentation of a single EEG channel, as proposed in [1]_.

    Function copied from https://github.com/cliffordlab/sleep-convolutions-tf
    and modified.

    Parameters
    ----------
    X : torch.Tensor
        EEG input example or batch.
    y : torch.Tensor
        EEG labels for the example or batch.
    phase_noise_magnitude: float
        Float between 0 and 1 setting the range over which the phase
        pertubation is uniformly sampled:
        [0, `phase_noise_magnitude` * 2 * `pi`].
    channel_indep : bool
        Whether to sample phase perturbations independently for each channel or
        not. It is advised to set it to False when spatial information is
        important for the task, like in BCI.
    random_state: int | numpy.random.Generator, optional
        Used to draw the phase perturbation. Defaults to None.

    Returns
    -------
    torch.Tensor
        Transformed inputs.
    torch.Tensor
        Transformed labels.

    References
    ----------
    .. [1] Schwabedal, J. T., Snyder, J. C., Cakmak, A., Nemati, S., &
       Clifford, G. D. (2018). Addressing Class Imbalance in Classification
       Problems of Noisy Signals by using Fourier Transform Surrogates. arXiv
       preprint arXiv:1806.08675.
    """
    assert isinstance(
        phase_noise_magnitude,
        (Real, torch.FloatTensor, torch.cuda.FloatTensor)
    ) and 0 <= phase_noise_magnitude <= 1, (
        f"eps must be a float beween 0 and 1. Got {phase_noise_magnitude}.")

    f = fft(X.double(), dim=-1)
    device = X.device

    n = f.shape[-1]
    random_phase = _new_random_fft_phase[n % 2](
        f.shape[0],
        f.shape[-2] if channel_indep else 1,
        n,
        device=device,
        random_state=random_state
    )
    if not channel_indep:
        random_phase = torch.tile(random_phase, (1, f.shape[-2], 1))
    if isinstance(phase_noise_magnitude, torch.Tensor):
        phase_noise_magnitude = phase_noise_magnitude.to(device)
    f_shifted = f * torch.exp(phase_noise_magnitude * random_phase)
    shifted = ifft(f_shifted, dim=-1)
    transformed_X = shifted.real.float()

    return transformed_X, y


def _pick_channels_randomly(X, p_pick, random_state):
    rng = check_random_state(random_state)
    batch_size, n_channels, _ = X.shape
    # allows to use the same RNG
    unif_samples = torch.as_tensor(
        rng.uniform(0, 1, size=(batch_size, n_channels)),
        dtype=torch.float,
        device=X.device,
    )
    # equivalent to a 0s and 1s mask
    return torch.sigmoid(1000 * (unif_samples - p_pick))


def channels_dropout(X, y, p_drop, random_state=None):
    """Randomly set channels to flat signal.

    Part of the CMSAugment policy proposed in [1]_

    Parameters
    ----------
    X : torch.Tensor
        EEG input example or batch.
    y : torch.Tensor
        EEG labels for the example or batch.
    p_drop : float
        Float between 0 and 1 setting the probability of dropping each channel.
    random_state : int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Defaults to None.

    Returns
    -------
    torch.Tensor
        Transformed inputs.
    torch.Tensor
        Transformed labels.

    References
    ----------
    .. [1] Saeed, A., Grangier, D., Pietquin, O., & Zeghidour, N. (2020).
       Learning from Heterogeneous EEG Signals with Differentiable Channel
       Reordering. arXiv preprint arXiv:2010.13694.
    """
    mask = _pick_channels_randomly(X, p_drop, random_state=random_state)
    return X * mask.unsqueeze(-1), y


def _make_permutation_matrix(X, mask, random_state):
    rng = check_random_state(random_state)
    batch_size, n_channels, _ = X.shape
    hard_mask = mask.round()
    batch_permutations = torch.empty(
        batch_size, n_channels, n_channels, device=X.device
    )
    for b, mask in enumerate(hard_mask):
        channels_to_shuffle = torch.arange(n_channels)
        channels_to_shuffle = channels_to_shuffle[mask.bool()]
        channels_permutation = np.arange(n_channels)
        channels_permutation[channels_to_shuffle] = rng.permutation(
            channels_to_shuffle
        )
        channels_permutation = torch.as_tensor(
            channels_permutation, dtype=torch.int64, device=X.device
        )
        batch_permutations[b, ...] = one_hot(channels_permutation)
    return batch_permutations


def channels_shuffle(X, y, p_shuffle, random_state=None):
    """Randomly shuffle channels in EEG data matrix.

    Part of the CMSAugment policy proposed in [1]_

    Parameters
    ----------
    X : torch.Tensor
        EEG input example or batch.
    y : torch.Tensor
        EEG labels for the example or batch.
    p_shuffle: float | None
        Float between 0 and 1 setting the probability of including the channel
        in the set of permutted channels.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Used to sample which channels to shuffle and to carry the shuffle.
        Defaults to None.

    Returns
    -------
    torch.Tensor
        Transformed inputs.
    torch.Tensor
        Transformed labels.

    References
    ----------
    .. [1] Saeed, A., Grangier, D., Pietquin, O., & Zeghidour, N. (2020).
       Learning from Heterogeneous EEG Signals with Differentiable Channel
       Reordering. arXiv preprint arXiv:2010.13694.
    """
    if p_shuffle == 0:
        return X, y
    mask = _pick_channels_randomly(X, 1 - p_shuffle, random_state)
    batch_permutations = _make_permutation_matrix(X, mask, random_state)
    return torch.matmul(batch_permutations, X), y


def gaussian_noise(X, y, std, random_state=None):
    """Randomly add white Gaussian noise to all channels.

    Suggested e.g. in [1]_, [2]_ and [3]_

    Parameters
    ----------
    X : torch.Tensor
        EEG input example or batch.
    y : torch.Tensor
        EEG labels for the example or batch.
    std : float
        Standard deviation to use for the additive noise.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Defaults to None.

    Returns
    -------
    torch.Tensor
        Transformed inputs.
    torch.Tensor
        Transformed labels.

    References
    ----------
    .. [1] Wang, F., Zhong, S. H., Peng, J., Jiang, J., & Liu, Y. (2018). Data
       augmentation for eeg-based emotion recognition with deep convolutional
       neural networks. In International Conference on Multimedia Modeling
       (pp. 82-93).
    .. [2] Cheng, J. Y., Goh, H., Dogrusoz, K., Tuzel, O., & Azemi, E. (2020).
       Subject-aware contrastive learning for biosignals. arXiv preprint
       arXiv:2007.04871.
    .. [3] Mohsenvand, M. N., Izadi, M. R., & Maes, P. (2020). Contrastive
       Representation Learning for Electroencephalogram Classification. In
       Machine Learning for Health (pp. 238-253). PMLR.
    """
    rng = check_random_state(random_state)
    if isinstance(std, torch.Tensor):
        std = std.to(X.device)
    noise = torch.from_numpy(
        rng.normal(
            loc=np.zeros(X.shape),
            scale=1
        ),
    ).float().to(X.device) * std
    transformed_X = X + noise
    return transformed_X, y


def smooth_time_mask(X, y, mask_start_per_sample, mask_len_samples):
    """Smoothly replace a contiguous part of all channels by zeros.

    Originally proposed in [1]_ and [2]_

    Parameters
    ----------
    X : torch.Tensor
        EEG input example or batch.
    y : torch.Tensor
        EEG labels for the example or batch.
    mask_start_per_sample : torch.tensor
        Tensor of integers containing the position (in last dimension) where to
        start masking the signal. Should have the same size as the first
        dimension of X (i.e. one start position per example in the batch).
    mask_len_samples : int
        Number of consecutive samples to zero out.

    Returns
    -------
    torch.Tensor
        Transformed inputs.
    torch.Tensor
        Transformed labels.

    References
    ----------
    .. [1] Cheng, J. Y., Goh, H., Dogrusoz, K., Tuzel, O., & Azemi, E. (2020).
       Subject-aware contrastive learning for biosignals. arXiv preprint
       arXiv:2007.04871.
    .. [2] Mohsenvand, M. N., Izadi, M. R., & Maes, P. (2020). Contrastive
       Representation Learning for Electroencephalogram Classification. In
       Machine Learning for Health (pp. 238-253). PMLR.
    """
    batch_size, n_channels, seq_len = X.shape
    t = torch.arange(seq_len, device=X.device).float()
    t = t.repeat(batch_size, n_channels, 1)
    mask_start_per_sample = mask_start_per_sample.view(-1, 1, 1)
    s = 1000 / seq_len
    mask = (torch.sigmoid(s * -(t - mask_start_per_sample)) +
            torch.sigmoid(s * (t - mask_start_per_sample - mask_len_samples))
            ).float().to(X.device)
    return X * mask, y


def _nextpow2(n):
    """Return the first integer N such that 2**N >= abs(n)."""
    return int(np.ceil(np.log2(np.abs(n))))


def _analytic_transform(x):
    if torch.is_complex(x):
        raise ValueError("x must be real.")

    N = x.shape[-1]
    f = fft(x, N, dim=-1)
    h = torch.zeros_like(f)
    if N % 2 == 0:
        h[..., 0] = h[..., N // 2] = 1
        h[..., 1:N // 2] = 2
    else:
        h[..., 0] = 1
        h[..., 1:(N + 1) // 2] = 2

    return ifft(f * h, dim=-1)


def _frequency_shift(X, fs, f_shift):
    """
    Shift the specified signal by the specified frequency.

    See https://gist.github.com/lebedov/4428122
    """
    # Pad the signal with zeros to prevent the FFT invoked by the transform
    # from slowing down the computation:
    n_channels, N_orig = X.shape[-2:]
    N_padded = 2 ** _nextpow2(N_orig)
    t = torch.arange(N_padded, device=X.device) / fs
    padded = pad(X, (0, N_padded - N_orig))
    analytical = _analytic_transform(padded)
    if isinstance(f_shift, (float, int, np.ndarray, list)):
        f_shift = torch.as_tensor(f_shift).float()
    reshaped_f_shift = f_shift.repeat(
        N_padded, n_channels, 1).T
    shifted = analytical * torch.exp(2j * np.pi * reshaped_f_shift * t)
    return shifted[..., :N_orig].real.float()


def frequency_shift(X, y, delta_freq, sfreq):
    """Adds a shift in the frequency domain to all channels.

    Note that here, the shift is the same for all channels of a single example.

    Parameters
    ----------
    X : torch.Tensor
        EEG input example or batch.
    y : torch.Tensor
        EEG labels for the example or batch.
    delta_freq : float
        The amplitude of the frequency shift (in Hz).
    sfreq : float
        Sampling frequency of the signals to be transformed.

    Returns
    -------
    torch.Tensor
        Transformed inputs.
    torch.Tensor
        Transformed labels.
    """
    transformed_X = _frequency_shift(
        X=X,
        fs=sfreq,
        f_shift=delta_freq,
    )
    return transformed_X, y


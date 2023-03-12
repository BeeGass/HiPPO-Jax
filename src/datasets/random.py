from src.datasets.dataset import Dataset
import torch
from torchvision import datasets, transforms
import numpy as np


class WhiteSignalDataset:
    def __init__(self, period, dt, freq, rms=0.5, batch_shape=(), num_samples=1000):
        self.period = period
        self.dt = dt
        self.freq = freq
        self.rms = rms
        self.batch_shape = batch_shape
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        signal = self.whitesignal(
            self.period, self.dt, self.freq, self.rms, self.batch_shape
        )
        return torch.from_numpy(signal.astype(np.float32))

    def whitesignal(self, period, dt, freq, rms=0.5, batch_shape=()):
        """
        Produces output signal of length period / dt, band-limited to frequency freq
        Output shape (*batch_shape, period/dt)
        Adapted from the nengo library
        """

        if freq is not None and freq < 1.0 / period:
            raise ValueError(
                f"Make ``{freq=} >= 1. / {period=}`` to produce a non-zero signal",
            )

        nyquist_cutoff = 0.5 / dt
        if freq > nyquist_cutoff:
            raise ValueError(
                f"{freq} must not exceed the Nyquist frequency for the given dt ({nyquist_cutoff:0.3f})"
            )

        n_coefficients = int(np.ceil(period / dt / 2.0))
        shape = batch_shape + (n_coefficients + 1,)
        sigma = rms * np.sqrt(0.5)
        coefficients = 1j * np.random.normal(0.0, sigma, size=shape)
        coefficients[..., -1] = 0.0
        coefficients += np.random.normal(0.0, sigma, size=shape)
        coefficients[..., 0] = 0.0

        set_to_zero = np.fft.rfftfreq(2 * n_coefficients, d=dt) > freq
        coefficients *= 1 - set_to_zero
        power_correction = np.sqrt(
            1.0 - np.sum(set_to_zero, dtype=float) / n_coefficients
        )
        if power_correction > 0.0:
            coefficients /= power_correction
        coefficients *= np.sqrt(2 * n_coefficients)
        signal = np.fft.irfft(coefficients, axis=-1)
        signal = signal - signal[..., :1]  # Start from 0
        return signal


class WhiteSignal(Dataset):
    def __init__(
        self,
        name: str,
        path: str,
        period,
        dt,
        freq,
        rms=0.5,
        num_samples=1000,
        batch_size: int = 64,
    ):
        super().__init__(name, path)
        self.batch_size = batch_size

        train_dataset = WhiteSignalDataset(
            period,
            dt,
            freq,
            rms,
            batch_shape=(self.batch_size,),
            num_samples=num_samples,
        )
        test_dataset = WhiteSignalDataset(
            period, dt, freq, rms, num_samples=num_samples
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=True
        )

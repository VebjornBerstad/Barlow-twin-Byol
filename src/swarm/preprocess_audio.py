import torch as T
import torchaudio as TA


def convert_to_mono(waveform: T.Tensor) -> T.Tensor:
    if waveform.ndim > 1:
        if waveform.ndim != 2:
            raise ValueError(f"Expected waveform to have 1 or 2 dimensions, but got {waveform.ndim}.")
        if waveform.shape[0] > 1:
            waveform = T.mean(waveform, dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
    return waveform


def convert_waveform_to_lms(
    waveform: T.Tensor,
    waveform_sample_rate: int,
    target_sample_rate: int,
    n_mels: int,
    f_min: int,
    f_max: int,
    device: T.device | str = 'cpu',
):
    waveform = waveform.to(device)
    # Load audio and resample to 16 kHz
    resampler = TA.transforms.Resample(orig_freq=waveform_sample_rate, new_freq=target_sample_rate).to(device)
    waveform = resampler(waveform)

    # Convert to mono.
    waveform = convert_to_mono(waveform)

    # Convert to mel-spectrogram
    n_fft = int(0.064 * target_sample_rate)  # 64 ms window
    hop_length = int(0.01 * target_sample_rate)  # 10 ms step size
    mel_transform = TA.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max
    ).to(device)

    mel_spectrogram = mel_transform(waveform)

    # TODO: Should this be mel_spectrogram.log() instead?
    # Convert to log-scaled mel-spectrogram
    amp_to_db = TA.transforms.AmplitudeToDB(stype='power').to(device)
    log_mel_spectrogram = amp_to_db(mel_spectrogram)

    return log_mel_spectrogram


def create_audio_segments(
    waveform: T.Tensor,
    sample_rate: int,
    segment_length_sec: float,
    hop_length_sec: float,
) -> list[T.Tensor]:
    # Ensure the waveform is mono.
    waveform = convert_to_mono(waveform)

    # Convert segment/hop lengths to samples.
    segment_length_samples = int(segment_length_sec * sample_rate)
    hop_length_samples = int(hop_length_sec * sample_rate)

    # TODO: Should we pad the waveform to ensure we don't lose any audio?

    # Create segments.
    segments = []
    for i in range(0, waveform.shape[-1] - segment_length_samples + 1, hop_length_samples):
        segment = waveform[..., i:i + segment_length_samples]
        segments.append(segment)

    return segments

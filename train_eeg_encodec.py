"""
Training and testing script for EEG‑to‑audio reconstruction using a modified
EnCodec model.  This script glues together a user‑provided EEG encoder with
the quantizer and decoder from a pretrained EnCodec checkpoint.  It
performs a simple train/test split on a custom EEG/audio dataset and
trains only the EEG encoder while keeping the quantizer and decoder
frozen.  Losses are computed using time domain L1 and a multi‑resolution
STFT loss similar to the original EnCodec training pipeline.

This script assumes the following prerequisites:

* A pretrained EnCodec checkpoint containing a quantizer and decoder.  The
  checkpoint can be the one obtained after the first stage of EnCodec
  training (e.g. from ``encodec-pytorch``).  Use the function
  ``EncodecModel.my_encodec_model`` from the provided code base to
  initialise the quantizer and decoder.  Alternatively you can load
  ``encodec_model_24khz`` and extract its quantizer/decoder.
* An EEG encoder module provided by the user.  The module should export
  a class (e.g. ``CNN_Mamba_v2``) which accepts a 3D tensor of shape
  ``[batch, channels, time]`` and returns latent features with shape
  ``[batch, latent_channels, time_code]`` compatible with the EnCodec
  quantizer.  You should ensure the output dimension matches the
  ``dimension`` field of the pretrained encoder in the checkpoint.
* A dataset consisting of paired EEG signals and their corresponding
  audio segments.  EEG signals are expected to be stored as 32‑channel
  recordings sampled at 500 Hz, while audio targets are mono or stereo
  waveforms sampled at 24 kHz.  Each EEG recording should align with a
  fixed‑duration audio clip (e.g. eight seconds).  For single‑song
  experiments, the same audio clip can be reused across all EEG
  segments.

Example usage:

```
python train_eeg_encodec.py \
    --eeg-dir /path/to/eeg_npy \
    --audio-path /path/to/song.wav \
    --checkpoint /path/to/encodec_checkpoint.pt \
    --epochs 100 \
    --batch-size 4 \
    --lr 1e-4 \
    --log-dir runs/eeg_experiment
```

The script will create a TensorBoard log directory under ``--log-dir`` and
write scalar losses during training and evaluation.  It will also
periodically save model checkpoints and a few reconstructed audio files
from the test set into ``--output-dir``.
"""

import argparse
import logging
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

# -----------------------------------------------------------------------------
# EEG encoder import
#
# The EEG encoder lives in the ``modules_v2`` folder of the encodec‑pytorch
# repository.  File names in that directory use hyphens (e.g. ``CNN-Mamba_v2.py``)
# which are not directly importable by Python's dotted module syntax.  To
# accommodate this, we attempt an import from a correctly named module first,
# then fall back to dynamically loading the file by path if necessary.

def _load_eeg_encoder_class() -> type:
    """Dynamically import the EEG encoder class from modules_v2.

    Returns
    -------
    type
        The class implementing the EEG encoder.  By convention this is
        ``EEGEncoderSEAMamba`` defined in ``modules_v2/CNN-Mamba_v2.py``.  If
        that fails, falls back to ``EEGEncoder`` defined in
        ``modules_v2/CNN-Mamba.py``.
    Raises
    ------
    ImportError
        If neither module can be loaded.
    """
    # Try the canonical import if the module has been renamed to a valid
    # Python identifier (e.g. CNN_Mamba_v2.py -> CNN_Mamba_v2)
    try:
        from modules_v2.CNN_Mamba_v2 import EEGEncoderSEAMamba  # type: ignore
        return EEGEncoderSEAMamba
    except Exception:
        pass
    # Try dynamic import from the hyphenated filename CNN-Mamba_v2.py
    import importlib.util
    import os
    import sys
    # Determine the directory of this script and build the path to modules_v2
    # relative to it.  When running inside the encodec repository, modules_v2
    # should reside at the same level as this script.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.join(base_dir, 'modules_v2', 'CNN_Mamba_v2.py'),
        os.path.join(base_dir, 'modules_v2', 'CNN_Mamba.py'),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location('eeg_encoder_module', path)
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                # Prefer the SEAMamba class if present
                if hasattr(module, 'EEGEncoderSEAMamba'):
                    return getattr(module, 'EEGEncoderSEAMamba')
                # Otherwise fall back to EEGEncoder
                if hasattr(module, 'EEGEncoder'):
                    return getattr(module, 'EEGEncoder')
    # If nothing worked, raise an error
    raise ImportError(
        'Unable to locate EEG encoder. Please ensure the modules_v2 directory '
        'is present and contains CNN-Mamba_v2.py or CNN-Mamba.py with an '
        'EEGEncoderSEAMamba or EEGEncoder class.'
    )

# Defer loading the encoder class until main() executes so that patching or
# modifications to sys.path take effect.


# Import EnCodec model factory to load pre‑trained quantizer and decoder.
try:
    from model import EncodecModel
except ImportError:
    raise ImportError(
        "Could not import EncodecModel. Make sure the encodec-pytorch "
        "repository is on the Python path."
    )

# Import the mel spectrogram helper used to compute the STFT loss.  The
# ``audio_to_mel.Audio2Mel`` class matches the one used in the original
# encodec repo.  It wraps ``torchaudio`` to produce mel spectrograms.
try:
    from audio_to_mel import Audio2Mel
except ImportError:
    # Fallback: define a simple mel spectrogram using torchaudio if
    # audio_to_mel is unavailable.  This will compute the mel spectrogram
    # directly via torchaudio.transforms.MelSpectrogram.
    class Audio2Mel(nn.Module):
        def __init__(self,
                     n_fft: int,
                     win_length: int,
                     hop_length: int,
                     n_mel_channels: int,
                     sampling_rate: int):
            super().__init__()
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=sampling_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                n_mels=n_mel_channels,
                f_min=0.0,
                f_max=float(sampling_rate // 2),
                power=2.0,
            )
            self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: [B, C, T]
            # Convert to mono if necessary by averaging channels
            if x.dim() == 3 and x.size(1) > 1:
                x_mono = x.mean(dim=1, keepdim=True)
            else:
                x_mono = x
            mel = self.mel(x_mono)  # [B, n_mel, frames]
            # Convert power spectrogram to log amplitude (optional)
            mel = self.amplitude_to_db(mel + 1e-10)
            return mel


class EEGAudioDataset(Dataset):
    """Dataset for paired EEG and audio segments.

    Each EEG file in ``eeg_dir`` corresponds to one recording of a subject
    listening to a fixed duration of music.  The audio clip is loaded
    once from ``audio_path`` and repeated for each EEG segment.

    EEG files are expected to be stored as numpy arrays (.npy) with
    shape [channels, time] and sampling rate 500 Hz.  Audio files are
    loaded via torchaudio and resampled to the target sample rate if
    necessary.
    """

    def __init__(self,
                 eeg_dir: str,
                 audio_path: str,
                 sample_rate: int = 24_000,
                 duration: float = 8.0):
        self.eeg_files: List[Path] = sorted(
            [Path(eeg_dir) / fname for fname in os.listdir(eeg_dir)
             if fname.lower().endswith('.npy')]
        )
        if not self.eeg_files:
            raise RuntimeError(f"No .npy files found in {eeg_dir}")
        self.sample_rate = sample_rate
        self.duration = duration
        # Load audio clip once and ensure its length matches the target duration
        audio, sr = torchaudio.load(audio_path)  # shape [C, T]
        if sr != sample_rate:
            audio = torchaudio.functional.resample(audio, sr, sample_rate)
        expected_length = int(duration * sample_rate)
        if audio.size(1) < expected_length:
            raise ValueError(
                f"Audio file {audio_path} is shorter ({audio.size(1)/sr:.2f}s) than "
                f"the required duration {duration}s"
            )
        # Trim or pad the audio to exactly duration seconds
        self.audio = audio[:, :expected_length]

    def __len__(self) -> int:
        return len(self.eeg_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        eeg_file = self.eeg_files[index]
        eeg_np = np.load(eeg_file).astype(np.float32)  # shape [C_eeg, T_eeg]
        eeg = torch.from_numpy(eeg_np)
        # Repeat audio for each EEG segment.  We clone so that autograd
        # operations do not link across samples.
        audio = self.audio.clone()
        return eeg, audio


class EEGEncodecModel(nn.Module):
    """Composite model mapping EEG to audio via quantization and decoding.

    Parameters
    ----------
    eeg_encoder : nn.Module
        A neural network mapping EEG signals to latent embeddings.  Its
        output should have shape [B, latent_dim, T_code].
    quantizer : callable
        A residual vector quantizer loaded from a pretrained EnCodec
        checkpoint.  It must accept arguments (embedding, frame_rate,
        bandwidth) and return an object with attributes ``quantized`` and
        ``penalty``.
    decoder : nn.Module
        The decoder network from a pretrained EnCodec model that maps
        quantized embeddings back to time‑domain audio.  Input shape
        should be [B, latent_dim, T_code].
    frame_rate : int
        The number of latent time steps per second produced by the EEG
        encoder.  This should match the frame rate used during
        quantizer training (e.g. 75 for 24 kHz audio when using ratios
        [8, 5, 4, 2]).
    bandwidth : float
        Target bandwidth in kbps for training.  Typical values include
        1.5, 3, 6, 12, or 24.
    """

    def __init__(self,
                 eeg_encoder: nn.Module,
                 quantizer,
                 decoder: nn.Module,
                 frame_rate: int,
                 bandwidth: float = 6.0):
        super().__init__()
        self.eeg_encoder = eeg_encoder
        self.quantizer = quantizer
        self.decoder = decoder
        self.frame_rate = frame_rate
        self.bandwidth = bandwidth
        # Freeze quantizer and decoder parameters
        for p in self.quantizer.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

    def forward(self, eeg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # eeg shape: [B, C_eeg, T_eeg]
        # Compute latent representation via EEG encoder
        emb = self.eeg_encoder(eeg)  # expected [B, latent_dim, T_code]
        # Quantize the latent representation; returns quantized embeddings
        qv = self.quantizer(emb, self.frame_rate, self.bandwidth)
        quantized = qv.quantized  # [B, latent_dim, T_code]
        penalty = qv.penalty      # scalar commitment loss
        # Decode back to waveform
        wav_hat = self.decoder(quantized)
        return wav_hat, penalty


def compute_loss(output: torch.Tensor,
                 target: torch.Tensor,
                 penalty: torch.Tensor,
                 sample_rate: int,
                 weight_time: float = 1.0,
                 weight_freq: float = 1.0,
                 weight_vq: float = 0.1) -> Tuple[torch.Tensor, dict]:
    """Compute a combined time/frequency domain loss with quantization penalty.

    The loss comprises:

    * Time‑domain L1 loss between the reconstructed and target audio.
    * Multi‑resolution STFT loss across a range of window sizes, each
      combining L1 and L2 differences on the mel spectrogram.
    * A quantization commitment penalty provided by the quantizer.

    Returns the scalar total loss and a dictionary of individual terms for
    logging.
    """
    l1_loss = torch.nn.functional.l1_loss(output, target)
    # Multi‑resolution STFT/Mel loss
    stft_loss = output.new_zeros(())
    # Use several scales ranging from 2^5 to 2^11 as in the encodec loss
    for i in range(5, 12):
        n_fft = 2 ** i
        win_length = n_fft
        hop_length = n_fft // 4
        # 64 mel channels as in the original code
        mel = Audio2Mel(n_fft=n_fft,
                        win_length=win_length,
                        hop_length=hop_length,
                        n_mel_channels=64,
                        sampling_rate=sample_rate).to(output.device)
        mel_output = mel(output)
        mel_target = mel(target)
        # Combine L1 and L2 losses on the mel spectrogram
        stft_loss = stft_loss + torch.nn.functional.l1_loss(mel_output, mel_target)
        stft_loss = stft_loss + torch.nn.functional.mse_loss(mel_output, mel_target)
    # Scale losses
    total = weight_time * l1_loss + weight_freq * stft_loss + weight_vq * penalty
    return total, {
        'l1': l1_loss.detach().cpu().item(),
        'stft': stft_loss.detach().cpu().item(),
        'vq_penalty': penalty.detach().cpu().item(),
        'total': total.detach().cpu().item(),
    }


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(
        model: EEGEncodecModel,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        sample_rate: int,
        writer: SummaryWriter,
        epoch: int,
        global_step: int,
        log_interval: int = 10) -> int:
    model.train()
    running_loss = 0.0
    for batch_idx, (eeg, audio) in enumerate(loader):
        eeg = eeg.to(device)
        audio = audio.to(device)
        optimizer.zero_grad()
        wav_hat, penalty = model(eeg)
        # Ensure the reconstructed waveform has the same length as the target
        # Some decoders produce slightly longer outputs; crop accordingly
        min_len = min(wav_hat.size(-1), audio.size(-1))
        wav_hat = wav_hat[..., :min_len]
        audio = audio[..., :min_len]
        loss, components = compute_loss(wav_hat, audio, penalty, sample_rate)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # Log to tensorboard
        writer.add_scalar('train/total_loss', components['total'], global_step)
        writer.add_scalar('train/l1', components['l1'], global_step)
        writer.add_scalar('train/stft', components['stft'], global_step)
        writer.add_scalar('train/vq_penalty', components['vq_penalty'], global_step)
        if (batch_idx + 1) % log_interval == 0:
            logging.info(
                f"Epoch {epoch} [{batch_idx + 1}/{len(loader)}] "
                f"Loss: {loss.item():.4f} (L1 {components['l1']:.4f}, "
                f"STFT {components['stft']:.4f}, VQ {components['vq_penalty']:.4f})"
            )
        global_step += 1
    avg_loss = running_loss / len(loader)
    writer.add_scalar('train/avg_loss', avg_loss, epoch)
    return global_step


def evaluate(
        model: EEGEncodecModel,
        loader: DataLoader,
        device: torch.device,
        sample_rate: int,
        writer: SummaryWriter,
        epoch: int,
        output_dir: Path) -> None:
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch_idx, (eeg, audio) in enumerate(loader):
            eeg = eeg.to(device)
            audio = audio.to(device)
            wav_hat, penalty = model(eeg)
            min_len = min(wav_hat.size(-1), audio.size(-1))
            wav_hat = wav_hat[..., :min_len]
            audio = audio[..., :min_len]
            loss, components = compute_loss(wav_hat, audio, penalty, sample_rate)
            running_loss += loss.item()
            # Log only first few examples as audio files
            if batch_idx < 5:
                # Save reconstructed and ground truth audio
                recon_path = output_dir / f"epoch{epoch}_sample{batch_idx}_recon.wav"
                target_path = output_dir / f"epoch{epoch}_sample{batch_idx}_target.wav"
                torchaudio.save(recon_path.as_posix(), wav_hat.cpu(), sample_rate)
                torchaudio.save(target_path.as_posix(), audio.cpu(), sample_rate)
            # Record metrics in TensorBoard
            global_index = epoch * len(loader) + batch_idx
            writer.add_scalar('eval/total_loss', components['total'], global_index)
            writer.add_scalar('eval/l1', components['l1'], global_index)
            writer.add_scalar('eval/stft', components['stft'], global_index)
            writer.add_scalar('eval/vq_penalty', components['vq_penalty'], global_index)
    avg_loss = running_loss / len(loader)
    writer.add_scalar('eval/avg_loss', avg_loss, epoch)
    logging.info(f"Evaluation epoch {epoch}: average loss {avg_loss:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EEG‑to‑audio model using EnCodec components.")
    parser.add_argument('--eeg-dir', type=str, required=True,
                        help='Directory containing .npy EEG recordings')
    parser.add_argument('--audio-path', type=str, required=True,
                        help='Path to the audio file (wav) used for all EEG segments')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the pretrained EnCodec checkpoint (.pt or .th)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Mini‑batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--bandwidth', type=float, default=6.0, help='Target bandwidth in kbps')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Proportion of data to use for training (remainder used for testing)')
    parser.add_argument('--log-dir', type=str, default='runs/eeg_train',
                        help='TensorBoard log directory')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Directory to save reconstructed audio examples and checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    set_random_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset and split into train/test sets
    dataset = EEGAudioDataset(args.eeg_dir, args.audio_path, sample_rate=24_000)
    n_total = len(dataset)
    n_train = int(n_total * args.train_ratio)
    n_test = n_total - n_train
    if n_train == 0 or n_test == 0:
        raise RuntimeError(
            f"Dataset too small for the requested train_ratio {args.train_ratio:.2f}. "
            f"Found {n_total} samples."
        )
    train_set, test_set = random_split(dataset, [n_train, n_test])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Load pretrained EnCodec model to extract quantizer and decoder
    if args.checkpoint.endswith('.pt') or args.checkpoint.endswith('.th'):
        # Use the helper from encodec model to load a custom checkpoint
        pretrained = EncodecModel.my_encodec_model(args.checkpoint)
    else:
        # Fallback to the default 24 kHz model
        pretrained = EncodecModel.encodec_model_24khz(pretrained=True)
    # Use the same frame rate as the pretrained model.  This ensures
    # compatibility with the quantizer and decoder.
    frame_rate = pretrained.frame_rate
    # Extract quantizer and decoder
    quantizer = pretrained.quantizer
    decoder = pretrained.decoder

    # Instantiate the user‑defined EEG encoder.  The class is loaded via
    # `_load_eeg_encoder_class()` to accommodate modules stored in
    # ``modules_v2/CNN-Mamba*.py``.  See the function definition above for
    # details.
    EEGEncoderClass = _load_eeg_encoder_class()
    eeg_encoder = EEGEncoderClass().to(device)

    # Build composite model
    model = EEGEncodecModel(
        eeg_encoder=eeg_encoder,
        quantizer=quantizer,
        decoder=decoder,
        frame_rate=frame_rate,
        bandwidth=args.bandwidth,
    ).to(device)

    # Optimizer (only EEG encoder parameters require gradients)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # Prepare logging
    writer = SummaryWriter(log_dir=args.log_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        logging.info(f"Starting epoch {epoch}/{args.epochs}")
        global_step = train_epoch(model, train_loader, optimizer, device, dataset.sample_rate, writer, epoch, global_step)
        evaluate(model, test_loader, device, dataset.sample_rate, writer, epoch, Path(args.output_dir))
        # Save checkpoint for this epoch
        ckpt_path = Path(args.output_dir) / f"eeg_encodec_epoch{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, ckpt_path)
        logging.info(f"Saved checkpoint to {ckpt_path}")

    writer.close()
    logging.info("Training complete.")


if __name__ == '__main__':
    main()
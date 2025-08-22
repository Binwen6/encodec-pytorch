import logging
import os
import warnings
from collections import defaultdict
import random
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import torchaudio
import numpy as np

import customAudioDataset as data
from customAudioDataset import collate_fn
from losses import disc_loss, total_loss
from model import EncodecModel
from msstftd import MultiScaleSTFTDiscriminator
from scheduler import WarmupCosineLrScheduler
from utils import (count_parameters, save_master_checkpoint, set_seed,
                   start_dist_train)
from balancer import Balancer

# -----------------------------------------------------------------------------
# EEG pair dataset and wrapper model
# -----------------------------------------------------------------------------
try:
    from modules_v2.CNN_Mamba_v2 import EEGEncoderSEAMamba  # type: ignore
except Exception:
    EEGEncoderSEAMamba = None  # will error later if eeg_pair is requested


class EEGAudioDataset(torch.utils.data.Dataset):
    """Dataset providing (EEG segment, target audio) pairs.

    Expects EEG segments saved as .npy shaped [32, 4000] at 500 Hz and a single
    reference audio clip to be repeated for each EEG segment. Audio is loaded
    once, resampled to the configured sample rate and trimmed/padded to exactly
    ``config.datasets.tensor_cut`` samples.
    """

    def __init__(self, config, mode: str = 'train'):
        self.eeg_dir = config.datasets.eeg_dir
        self.sample_rate = int(config.model.sample_rate)
        self.expected_audio_len = int(config.datasets.tensor_cut)
        self.rng = np.random.RandomState(config.common.seed)

        # Enumerate EEG npy files deterministically
        self.eeg_files = sorted([
            os.path.join(self.eeg_dir, f)
            for f in os.listdir(self.eeg_dir)
            if f.lower().endswith('.npy')
        ])
        if not self.eeg_files:
            raise RuntimeError(f"No .npy files found in {self.eeg_dir}")

        # Load audio once
        audio_path = config.datasets.audio_path
        wav, sr = torchaudio.load(audio_path)  # [C, T]
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        # Ensure mono
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        # Trim/pad to exact length
        if wav.size(1) < self.expected_audio_len:
            pad = self.expected_audio_len - wav.size(1)
            wav = torch.nn.functional.pad(wav, (0, pad))
        elif wav.size(1) > self.expected_audio_len:
            wav = wav[:, : self.expected_audio_len]
        self.audio = wav.contiguous()  # [1, expected_len]

    def __len__(self):
        return len(self.eeg_files)

    def __getitem__(self, index: int):
        eeg_np = np.load(self.eeg_files[index]).astype(np.float32)  # [32, 4000]
        eeg = torch.from_numpy(eeg_np)  # [C_eeg, T_eeg]
        target = self.audio.clone()     # [1, T]
        return eeg, target


def eeg_collate_fn(batch):
    eegs, targets = zip(*batch)
    eeg = torch.stack(eegs, dim=0).contiguous()        # [B, 32, 4000]
    target = torch.stack(targets, dim=0).contiguous()  # [B, 1, T]
    return eeg, target


class EEGEncodecWrapper(torch.nn.Module):
    """Wrap an EEG encoder with EnCodec quantizer and decoder.

    Returns output waveform and an auxiliary loss term (here, VQ penalty scaled)
    to be summed into the original training objective.
    """

    def __init__(self, base_encodec_model, frame_rate: int, target_frames: int, vq_weight: float = 0.1):
        super().__init__()
        if EEGEncoderSEAMamba is None:
            raise ImportError("EEGEncoderSEAMamba not available. Please ensure modules_v2/CNN_Mamba_v2.py exists.")
        # Use provided EnCodec components
        self.quantizer = base_encodec_model.quantizer
        self.decoder = base_encodec_model.decoder
        # Freeze quantizer and decoder
        for p in self.quantizer.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False
        # Ensure frozen modules are in eval mode (no running stat/codebook updates)
        self.quantizer.eval()
        self.decoder.eval()
        # Build EEG encoder; align latent time with decoder expectations
        self.eeg_encoder = EEGEncoderSEAMamba(target_frames=target_frames)
        self.frame_rate = int(frame_rate)
        self.vq_weight = float(vq_weight)

    def train(self, mode: bool = True):
        # Keep parent behavior for train/eval switching, but force frozen parts to eval
        super().train(mode)
        self.quantizer.eval()
        self.decoder.eval()
        return self

    def forward(self, eeg: torch.Tensor):
        # eeg: [B, 32, 4000]
        emb = self.eeg_encoder(eeg)  # [B, D, T_code]
        qv = self.quantizer(emb, self.frame_rate, max(self.decoder.bandwidths) if hasattr(self.decoder, 'bandwidths') else 6.0)
        quantized = qv.quantized
        # Straight-through estimator to preserve gradient flow to EEG encoder even if quantizer is eval/frozen
        quantized_st = quantized + (emb - emb.detach())
        wav_hat = self.decoder(quantized_st)
        aux = { 'vq_penalty': qv.penalty }
        penalty = qv.penalty
        # If penalty doesn't carry gradients (e.g., quantizer in eval), fall back to a commitment loss surrogate
        if not isinstance(penalty, torch.Tensor) or not penalty.requires_grad:
            penalty = torch.mean((emb - quantized.detach()) ** 2)
            aux['commitment_surrogate'] = penalty
        loss_w = self.vq_weight * penalty
        return wav_hat, loss_w, aux

warnings.filterwarnings("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define train one step function
def train_one_step(epoch,optimizer,optimizer_disc, model, disc_model, trainloader,config,scheduler,disc_scheduler,scaler=None,scaler_disc=None,writer=None,balancer=None):
    """train one step function

    Args:
        epoch (int): current epoch
        optimizer (_type_) : generator optimizer
        optimizer_disc (_type_): discriminator optimizer
        model (_type_): generator model
        disc_model (_type_): discriminator model
        trainloader (_type_): train dataloader
        config (_type_): hydra config file
        scheduler (_type_): adjust generate model learning rate
        disc_scheduler (_type_): adjust discriminator model learning rate
        warmup_scheduler (_type_): warmup learning rate
    """
    model.train()
    disc_model.train()
    data_length=len(trainloader)
    # Initialize variables to accumulate losses  
    accumulated_loss_g = 0.0
    accumulated_losses_g = defaultdict(float)
    accumulated_loss_w = 0.0
    accumulated_loss_disc = 0.0

    is_eeg_pair = hasattr(config.datasets, 'type') and str(config.datasets.type) == 'eeg_pair'
    for idx,batch in enumerate(trainloader):
        # warmup learning rate, warmup_epoch is defined in config file,default is 5
        if is_eeg_pair:
            eeg, target_wav = batch
            eeg = eeg.contiguous().cuda()            # [B, 32, 4000]
            input_wav = target_wav.contiguous().cuda()  # [B, 1, T]
        else:
            input_wav = batch.contiguous().cuda() #[B, 1, T]: eg. [2, 1, 203760]
        optimizer.zero_grad()
        with autocast(enabled=config.common.amp):
            if is_eeg_pair:
                output, loss_w, _ = model(eeg) # EEG -> wav
            else:
                output, loss_w, _ = model(input_wav) # wav -> wav
            logits_real, fmap_real = disc_model(input_wav)
            logits_fake, fmap_fake = disc_model(output)
            losses_g = total_loss(
                fmap_real, 
                logits_fake, 
                fmap_fake, 
                input_wav, 
                output, 
                sample_rate=config.model.sample_rate,
            ) 
        if config.common.amp: 
            loss = 3*losses_g['l_g'] + 3*losses_g['l_feat'] + losses_g['l_t']/10 + losses_g['l_f']  + loss_w
            # not implementing loss balancer in this section, since they say amp is not working anyway:
            # https://github.com/ZhikangNiu/encodec-pytorch/issues/21#issuecomment-2122593367
            scaler.scale(loss).backward()  
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  
            scaler.step(optimizer)  
            scaler.update()   
            # BUG: doesn't this get done later anyway?
            scheduler.step()  
        else:
            # They say they use multiple backwards calls, and lambda_w is 1...
            # https://github.com/facebookresearch/encodec/issues/20
            if balancer is not None:
                balancer.backward(losses_g, output, retain_graph=True)
                # naive loss summation for metrics below
                loss_g = sum([l * balancer.weights[k] for k, l in losses_g.items()])
            else:
                # without balancer: loss = 3*l_g + 3*l_feat + (l_t / 10) + l_f
                # loss_g = torch.tensor([0.0], device='cuda', requires_grad=True)
                loss_g = 3*losses_g['l_g'] + 3*losses_g['l_feat'] + losses_g['l_t']/10 + losses_g['l_f'] 
                loss_g.backward()
            loss_w.backward()
            optimizer.step()

        # Accumulate losses  
        accumulated_loss_g += loss_g.item()
        for k, l in losses_g.items():
            accumulated_losses_g[k] += l.item()
        accumulated_loss_w += loss_w.item()

        # only update discriminator with probability from paper (configure)
        optimizer_disc.zero_grad()
        # support boolean/float/fraction-string (e.g., "2/3") for train_discriminator
        td_value = config.model.train_discriminator
        if isinstance(td_value, bool):
            td_prob = 1.0 if td_value else 0.0
        elif isinstance(td_value, (int, float)):
            td_prob = float(td_value)
        elif isinstance(td_value, str):
            s = td_value.strip()
            if '/' in s:
                num, den = s.split('/', 1)
                td_prob = float(num) / float(den)
            else:
                td_prob = float(s)
        else:
            raise TypeError(f"Unsupported type for model.train_discriminator: {type(td_value)}")

        train_disc_flag = (
            (td_prob > 0.0)
            and (epoch >= config.lr_scheduler.warmup_epoch)
            and (random.random() < td_prob)
        )
        train_discriminator = torch.tensor([train_disc_flag], dtype=torch.bool, device='cuda')
        # fix https://github.com/ZhikangNiu/encodec-pytorch/issues/30
        if dist.is_initialized():
            dist.broadcast(train_discriminator, 0)

        if train_discriminator.item():
            with autocast(enabled=config.common.amp):
                logits_real, _ = disc_model(input_wav)
                logits_fake, _ = disc_model(output.detach()) # detach to avoid backpropagation to model
                loss_disc = disc_loss(logits_real, logits_fake) # compute discriminator loss
            if config.common.amp: 
                scaler_disc.scale(loss_disc).backward()
                # torch.nn.utils.clip_grad_norm_(disc_model.parameters(), 1.0)    
                scaler_disc.step(optimizer_disc)  
                scaler_disc.update()  
            else:
                loss_disc.backward() 
                optimizer_disc.step()

            # Accumulate discriminator loss  
            accumulated_loss_disc += loss_disc.item()
        scheduler.step()
        disc_scheduler.step()

        if (not config.distributed.data_parallel or dist.get_rank() == 0) and (idx % config.common.log_interval == 0 or idx == data_length - 1): 
            log_msg = (  
                f"Epoch {epoch} {idx+1}/{data_length}\tAvg loss_G: {accumulated_loss_g / (idx + 1):.4f}\tAvg loss_W: {accumulated_loss_w / (idx + 1):.4f}\tlr_G: {optimizer.param_groups[0]['lr']:.6e}\tlr_D: {optimizer_disc.param_groups[0]['lr']:.6e}\t"  
            ) 
            writer.add_scalar('Train/Loss_G', accumulated_loss_g / (idx + 1), (epoch-1) * len(trainloader) + idx)  
            for k, l in accumulated_losses_g.items():
                writer.add_scalar(f'Train/{k}', l / (idx + 1), (epoch-1) * len(trainloader) + idx)
            writer.add_scalar('Train/Loss_W', accumulated_loss_w / (idx + 1), (epoch-1) * len(trainloader) + idx) 
            if config.model.train_discriminator and epoch >= config.lr_scheduler.warmup_epoch:
                log_msg += f"loss_disc: {accumulated_loss_disc / (idx + 1) :.4f}"  
                writer.add_scalar('Train/Loss_Disc', accumulated_loss_disc / (idx + 1), (epoch-1) * len(trainloader) + idx) 
            logger.info(log_msg) 

@torch.no_grad()
def test(epoch, model, disc_model, testloader, config, writer):
    model.eval()
    is_eeg_pair = hasattr(config.datasets, 'type') and str(config.datasets.type) == 'eeg_pair'
    for idx, batch in enumerate(testloader):
        if is_eeg_pair:
            eeg, target = batch
            eeg = eeg.cuda()
            input_wav = target.cuda()
            output, _, _ = model(eeg)
        else:
            input_wav = batch.cuda()
            _out = model(input_wav)
            output = _out[0] if isinstance(_out, tuple) else _out
        logits_real, fmap_real = disc_model(input_wav)
        logits_fake, fmap_fake = disc_model(output)
        loss_disc = disc_loss(logits_real, logits_fake) # compute discriminator loss
        losses_g = total_loss(fmap_real, logits_fake, fmap_fake, input_wav, output) 

    if not config.distributed.data_parallel or dist.get_rank()==0:
        log_msg = (f'| TEST | epoch: {epoch} | loss_g: {sum([l.item() for l in losses_g.values()])} | loss_disc: {loss_disc.item():.4f}') 
        for k, l in losses_g.items():
            writer.add_scalar(f'Test/{k}', l.item(), epoch)  
        writer.add_scalar('Test/Loss_Disc', loss_disc.item(), epoch)
        logger.info(log_msg)

        # save a sample reconstruction with a safe max length to avoid OOM (no AMP)
        if is_eeg_pair:
            eeg, input_wav = testloader.dataset[0]
            eeg = eeg.cuda().unsqueeze(0)
            input_wav = input_wav.cuda()
            _out = model(eeg)
            output = (_out[0] if isinstance(_out, tuple) else _out).squeeze(0)
        else:
            input_wav, _ = testloader.dataset.get()
            input_wav = input_wav.cuda()
            _out = model(input_wav.unsqueeze(0))
            output = (_out[0] if isinstance(_out, tuple) else _out).squeeze(0)
        max_demo_len = (
            # config.datasets.tensor_cut
            240000
            if hasattr(config.datasets, 'tensor_cut') and config.datasets.tensor_cut and config.datasets.tensor_cut > 0
            # else 72000
            else 240000
        )
        demo_wav = input_wav[..., : min(input_wav.shape[-1], max_demo_len)]
        # summarywriter can't log stereo files 😅 so just save examples
        sp = Path(config.checkpoint.save_folder)
        torchaudio.save(sp/f'GT.wav', input_wav.cpu(), config.model.sample_rate)
        torchaudio.save(sp/f'Reconstruction.wav', output.cpu(), config.model.sample_rate)

def train(local_rank,world_size,config,tmp_file=None):
    """train main function."""
    # remove the logging handler "somebody" added
    logger.handlers.clear()

    # set logger
    file_handler = logging.FileHandler(f"{config.checkpoint.save_folder}/train_encodec_bs{config.datasets.batch_size}_lr{config.optimization.lr}.log")
    formatter = logging.Formatter('%(asctime)s: %(levelname)s: [%(filename)s: %(lineno)d]: %(message)s')
    file_handler.setFormatter(formatter)

    # print to screen
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    # add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # set seed
    if config.common.seed is not None:
        set_seed(config.common.seed)

    # set train dataset
    use_eeg_pair = hasattr(config.datasets, 'type') and str(config.datasets.type) == 'eeg_pair'
    if use_eeg_pair:
        full = EEGAudioDataset(config=config)
        n_total = len(full)
        n_train = int(n_total * float(getattr(config.datasets, 'train_ratio', 0.8)))
        n_test = n_total - n_train
        generator = torch.Generator().manual_seed(config.common.seed)
        trainset, testset = torch.utils.data.random_split(full, [n_train, n_test], generator=generator)
    else:
        trainset = data.CustomAudioDataset(config=config)
        testset = data.CustomAudioDataset(config=config,mode='test')
    # set encodec model and discriminator model
    base_model = EncodecModel._get_model(
        config.model.target_bandwidths, 
        config.model.sample_rate, 
        config.model.channels,
        causal=config.model.causal, model_norm=config.model.norm, 
        audio_normalize=config.model.audio_normalize,
        segment=eval(config.model.segment), name=config.model.name,
        ratios=config.model.ratios,
    )
    # Optionally load pretrained weights (e.g., quantizer/decoder) from checkpoint
    if use_eeg_pair and hasattr(config.checkpoint, 'checkpoint_path') and config.checkpoint.checkpoint_path:
        try:
            ckpt = torch.load(config.checkpoint.checkpoint_path, map_location='cpu')
            sd = ckpt.get('model_state_dict', ckpt)
            model_sd = base_model.state_dict()
            # Filter to matching keys and shapes, prefer quantizer/decoder blocks
            filtered = {}
            for k, v in sd.items():
                if k not in model_sd:
                    continue
                if v.shape != model_sd[k].shape:
                    continue
                if k.startswith('quantizer') or k.startswith('decoder'):
                    filtered[k] = v
            # Fallback: if filtered is too small, allow loading any matching keys
            if len(filtered) < 10:
                for k, v in sd.items():
                    if k in model_sd and v.shape == model_sd[k].shape and k not in filtered:
                        filtered[k] = v
            missing, unexpected = base_model.load_state_dict(filtered, strict=False)
            logger.info(f"Loaded {len(filtered)} weights from checkpoint into base model. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint '{config.checkpoint.checkpoint_path}': {e}")
    if use_eeg_pair:
        # Wrap EEG encoder with pretrained quantizer/decoder
        frame_rate = base_model.frame_rate if hasattr(base_model, 'frame_rate') else 75
        target_frames = int((config.datasets.tensor_cut / config.model.sample_rate) * frame_rate)
        model = EEGEncodecWrapper(base_model, frame_rate=frame_rate, target_frames=target_frames)
    else:
        model = base_model
    disc_model = MultiScaleSTFTDiscriminator(
        in_channels=config.model.channels,
        out_channels=config.model.channels,
        filters=config.model.filters,
        hop_lengths=config.model.disc_hop_lengths,
        win_lengths=config.model.disc_win_lengths,
        n_ffts=config.model.disc_n_ffts,
    )

    # log model, disc model parameters and train mode
    logger.info(model)
    logger.info(disc_model)
    logger.info(config)
    # Explicitly report whether real Mamba is used
    if use_eeg_pair and hasattr(model, 'eeg_encoder') and hasattr(model.eeg_encoder, 'mamba'):
        mamba_module = model.eeg_encoder.mamba.__class__.__module__
        logger.info(f"EEG Mamba implementation: {mamba_module} | using_mamba={( 'mamba_ssm' in mamba_module )}")
    logger.info(f"Encodec Model Parameters: {count_parameters(model)} | Disc Model Parameters: {count_parameters(disc_model)}")
    logger.info(f"model train mode :{model.training} | quantizer train mode :{model.quantizer.training} ")

    # resume training
    resume_epoch = 0
    if config.checkpoint.resume:
        # check the checkpoint_path
        assert config.checkpoint.checkpoint_path != '', "resume path is empty"
        assert config.checkpoint.disc_checkpoint_path != '', "disc resume path is empty"

        model_checkpoint = torch.load(config.checkpoint.checkpoint_path, map_location='cpu')
        disc_model_checkpoint = torch.load(config.checkpoint.disc_checkpoint_path, map_location='cpu')
        model.load_state_dict(model_checkpoint['model_state_dict'])
        disc_model.load_state_dict(disc_model_checkpoint['model_state_dict'])
        resume_epoch = model_checkpoint['epoch']
        if resume_epoch >= config.common.max_epoch:
            raise ValueError(f"resume epoch {resume_epoch} is larger than total epochs {config.common.epochs}")
        logger.info(f"load chenckpoint of model and disc_model, resume from {resume_epoch}")

    train_sampler = None
    test_sampler = None
    if config.distributed.data_parallel:
        # distributed init
        if config.distributed.init_method == "tmp":
            torch.distributed.init_process_group(
                backend='nccl',
                init_method="file://{}".format(tmp_file),
                rank=local_rank,
                world_size=world_size)
        elif config.distributed.init_method == "tcp":
            if "MASTER_ADDR" in os.environ:
                master_addr = os.environ['MASTER_ADDR']
            else:
                master_addr = "localhost"
            if "MASTER_PORT" in os.environ:
                master_port = os.environ["MASTER_PORT"]
            else:
                master_port = 6008

            distributed_init_method = "tcp://%s:%s" % (master_addr, master_port)
            logger.info(f"distributed_init_method : {distributed_init_method}")
            torch.distributed.init_process_group(
                backend='nccl',
                init_method=distributed_init_method,
                rank=local_rank,
                world_size=world_size)

        torch.cuda.set_device(local_rank) 
        torch.cuda.empty_cache()
        # set distributed sampler
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset)

    model.cuda()
    disc_model.cuda()

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=config.datasets.batch_size,
        sampler=train_sampler, 
        shuffle=(train_sampler is None), collate_fn=(eeg_collate_fn if use_eeg_pair else collate_fn),
        pin_memory=config.datasets.pin_memory)
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=config.datasets.batch_size,
        sampler=test_sampler, 
        shuffle=False, collate_fn=(eeg_collate_fn if use_eeg_pair else collate_fn),
        pin_memory=config.datasets.pin_memory)
    logger.info(f"There are {len(trainloader)} data to train the EnCodec")
    logger.info(f"There are {len(testloader)} data to test the EnCodec")

    # set optimizer and scheduler, warmup scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    disc_params = [p for p in disc_model.parameters() if p.requires_grad]
    optimizer = optim.Adam([{'params': params, 'lr': config.optimization.lr}], betas=(0.5, 0.9))
    optimizer_disc = optim.Adam([{'params':disc_params, 'lr': config.optimization.disc_lr}], betas=(0.5, 0.9))
    scheduler = WarmupCosineLrScheduler(optimizer, max_iter=config.common.max_epoch*len(trainloader), eta_ratio=0.1, warmup_iter=config.lr_scheduler.warmup_epoch*len(trainloader), warmup_ratio=1e-4)
    disc_scheduler = WarmupCosineLrScheduler(optimizer_disc, max_iter=config.common.max_epoch*len(trainloader), eta_ratio=0.1, warmup_iter=config.lr_scheduler.warmup_epoch*len(trainloader), warmup_ratio=1e-4)

    scaler = GradScaler() if config.common.amp else None
    scaler_disc = GradScaler() if config.common.amp else None  

    if config.checkpoint.resume and 'scheduler_state_dict' in model_checkpoint.keys() and 'scheduler_state_dict' in disc_model_checkpoint.keys(): 
        optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(model_checkpoint['scheduler_state_dict'])
        optimizer_disc.load_state_dict(disc_model_checkpoint['optimizer_state_dict'])
        disc_scheduler.load_state_dict(disc_model_checkpoint['scheduler_state_dict'])
        logger.info(f"load optimizer and disc_optimizer state_dict from {resume_epoch}")

    if config.distributed.data_parallel:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        disc_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(disc_model)
        # wrap the model by using DDP
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=config.distributed.find_unused_parameters)
        disc_model = torch.nn.parallel.DistributedDataParallel(
            disc_model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=config.distributed.find_unused_parameters)
    if not config.distributed.data_parallel or dist.get_rank() == 0:  
        # Prefer explicit tensorboard.log_dir if provided; fallback to previous behavior
        tb_log_dir = (
            config.tensorboard.log_dir if hasattr(config, 'tensorboard') and hasattr(config.tensorboard, 'log_dir') and config.tensorboard.log_dir
            else f'{config.checkpoint.save_folder}/runs'
        )
        writer = SummaryWriter(log_dir=tb_log_dir)  
        logger.info(f'Saving tensorboard logs to {Path(writer.log_dir).resolve()}')
    else:  
        writer = None  
    start_epoch = max(1,resume_epoch+1) # start epoch is 1 if not resume
    # instantiate loss balancer
    balancer = Balancer(dict(config.balancer.weights)) if hasattr(config, 'balancer') else None
    if balancer:
        logger.info(f'Loss balancer with weights {balancer.weights} instantiated')
    test(0, model, disc_model, testloader, config, writer)
    for epoch in range(start_epoch, config.common.max_epoch+1):
        train_one_step(
            epoch, optimizer, optimizer_disc, 
            model, disc_model, trainloader,config,
            scheduler,disc_scheduler,scaler,scaler_disc,writer,balancer)
        if epoch % config.common.test_interval == 0:
            test(epoch,model,disc_model,testloader,config,writer)
        # save checkpoint and epoch
        if epoch % config.common.save_interval == 0:
            model_to_save = model.module if config.distributed.data_parallel else model
            disc_model_to_save = disc_model.module if config.distributed.data_parallel else disc_model 
            if not config.distributed.data_parallel or dist.get_rank() == 0:  
                save_master_checkpoint(epoch, model_to_save, optimizer, scheduler, f'{config.checkpoint.save_location}epoch{epoch}_lr{config.optimization.lr}.pt')  
                save_master_checkpoint(epoch, disc_model_to_save, optimizer_disc, disc_scheduler, f'{config.checkpoint.save_location}epoch{epoch}_disc_lr{config.optimization.lr}.pt') 

    if config.distributed.data_parallel:
        dist.destroy_process_group()

@hydra.main(config_path='config', config_name='config')
def main(config):
    # set distributed debug, if you encouter some multi gpu bug, please set torch_distributed_debug=True
    if config.distributed.torch_distributed_debug: 
        os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
        os.environ["TORCH_DISTRIBUTED_DEBUG"]="DETAIL"
    if not os.path.exists(config.checkpoint.save_folder):
        os.makedirs(config.checkpoint.save_folder)
    # disable cudnn
    torch.backends.cudnn.enabled = False
    # set distributed
    if config.distributed.data_parallel:  
        world_size = config.distributed.world_size  
        if config.distributed.init_method == "tmp":  
            import tempfile  
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:  
                start_dist_train(train, world_size, config, tmp_file.name)  
        elif config.distributed.init_method == "tcp":  
            start_dist_train(train, world_size, config)  
    else:  
        train(1, 1, config)  # set single gpu train 


if __name__ == '__main__':
    main()

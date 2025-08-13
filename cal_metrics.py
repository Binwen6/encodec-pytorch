# core codes are copy from https://github.com/yangdongchao/AcademiCodec/tree/master/evaluation_metric/calculate_voc_obj_metrics/metrics
import argparse
import os
from pathlib import Path
import traceback

import librosa
import numpy as np
from pesq import cypesq, pesq
from pystoi import stoi
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(description="Compute STOI and PESQ measure")
    parser.add_argument(
        '-r',
        '--ref_dir',
        required=True,
        help="Reference wave folder."
    )
    parser.add_argument(
        '-d',
        '--deg_dir',
        required=True,
        help="Degraded wave folder."
    )
    parser.add_argument(
        '-s',
        '--sr',
        type=int,
        default=16000,
        help="encodec sample rate."
    )
    parser.add_argument(
        '-b',
        '--bandwidth',
        type=float,
        default=6,
        help="encodec bandwidth.",
    )
    parser.add_argument(
        '-e',
        "--ext",
        default="wav",
        type=str,
        help="file extension"
    )
    parser.add_argument(
        "-o",
        "--output_result_path",
        default="./results/",
        type=Path
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed debug info during metric computation",
    )
    return parser


def calculate_stoi(ref_wav, deg_wav, sr):
    """Calculate STOI score between ref_wav and deg_wav"""
    min_len = min(len(ref_wav), len(deg_wav))
    ref_wav = ref_wav[:min_len]
    deg_wav = deg_wav[:min_len]
    stoi_score = stoi(ref_wav, deg_wav, sr, extended=False)
    return stoi_score

def calculate_pesq(ref_wav, deg_wav, orig_sr, debug=False):
    """Calculate PESQ scores (NB/WB) with proper resampling.

    - WB requires 16 kHz input and sr=16000
    - NB requires 8 kHz input and sr=8000
    """
    min_len = min(len(ref_wav), len(deg_wav))
    ref_wav = ref_wav[:min_len]
    deg_wav = deg_wav[:min_len]

    # Wideband (16 kHz)
    if orig_sr != 16000:
        ref_wb = librosa.resample(ref_wav, orig_sr=orig_sr, target_sr=16000)
        deg_wb = librosa.resample(deg_wav, orig_sr=orig_sr, target_sr=16000)
    else:
        ref_wb, deg_wb = ref_wav, deg_wav
    # Narrowband (8 kHz)
    if orig_sr != 8000:
        ref_nb = librosa.resample(ref_wav, orig_sr=orig_sr, target_sr=8000)
        deg_nb = librosa.resample(deg_wav, orig_sr=orig_sr, target_sr=8000)
    else:
        ref_nb, deg_nb = ref_wav, deg_wav

    # Sanitize: finite, clip to [-1, 1], float32, ensure minimum durations
    def sanitize(x):
        x = np.nan_to_num(x, copy=False)
        max_abs = np.max(np.abs(x)) if x.size > 0 else 0.0
        if max_abs > 1.0:
            x = x / max_abs
        return x.astype(np.float32, copy=False)

    ref_wb = sanitize(ref_wb)
    deg_wb = sanitize(deg_wb)
    ref_nb = sanitize(ref_nb)
    deg_nb = sanitize(deg_nb)

    min_wb_len = int(0.25 * 16000)
    min_nb_len = int(0.25 * 8000)

    def _avg_chunked_pesq(a, b, sr, mode, win_sec=30):
        # Split long arrays into windows to avoid C-extension stack issues
        win = int(win_sec * sr)
        if len(a) < win:
            try:
                return pesq(sr, a, b, mode)
            except Exception:
                return None
        scores = []
        num_chunks = int(np.ceil(len(a) / win))
        for i in range(num_chunks):
            s = i * win
            e = min((i + 1) * win, len(a))
            try:
                scores.append(pesq(sr, a[s:e], b[s:e], mode))
            except Exception as ex:
                if debug:
                    print(f"  PESQ chunk {i+1}/{num_chunks} failed ({mode}): {ex}")
        if not scores:
            return None
        return float(np.mean(scores))

    wb_pesq_score = None
    nb_pesq_score = None
    if len(ref_wb) >= min_wb_len and len(deg_wb) >= min_wb_len:
        wb_pesq_score = _avg_chunked_pesq(ref_wb, deg_wb, 16000, 'wb')
    if len(ref_nb) >= min_nb_len and len(deg_nb) >= min_nb_len:
        nb_pesq_score = _avg_chunked_pesq(ref_nb, deg_nb, 8000, 'nb')

    if nb_pesq_score is None:
        nb_pesq_score = 0
    if wb_pesq_score is None:
        wb_pesq_score = 0

    if debug:
        info = {
            "orig_sr": orig_sr,
            "len_ref": int(len(ref_wav)),
            "len_deg": int(len(deg_wav)),
            "len_ref_wb": int(len(ref_wb)),
            "len_deg_wb": int(len(deg_wb)),
            "len_ref_nb": int(len(ref_nb)),
            "len_deg_nb": int(len(deg_nb)),
            "min_wb_len": int(min_wb_len),
            "min_nb_len": int(min_nb_len),
            "ref_max_abs": float(np.max(np.abs(ref_wav))) if ref_wav.size else 0.0,
            "deg_max_abs": float(np.max(np.abs(deg_wav))) if deg_wav.size else 0.0,
            "ref_nan": int(np.isnan(ref_wav).sum()) if ref_wav.size else 0,
            "deg_nan": int(np.isnan(deg_wav).sum()) if deg_wav.size else 0,
            "nb_pesq": float(nb_pesq_score),
            "wb_pesq": float(wb_pesq_score),
        }
    else:
        info = None

    return nb_pesq_score, wb_pesq_score, info

def calculate_visqol_moslqo_score(ref_wav,deg_wav,mode='audio'):
    """Perceptual Quality Estimator for speech and audio
    you need to follow https://github.com/google/visqol to build & install 

    Args:
        ref_wav (_type_): re
        deg_wav (_type_): _description_
        mode (str, optional): _description_. Defaults to 'audio'.
    """
    try:
        from visqol import visqol_lib_py
        from visqol.pb2 import similarity_result_pb2, visqol_config_pb2
    except ImportError:
        print("visqol is not installed, please build and install follow https://github.com/google/visqol")
        
    config = visqol_config_pb2.VisqolConfig()

    if mode == "audio":
        config.audio.sample_rate = 48000
        config.options.use_speech_scoring = False
        svr_model_path = "libsvm_nu_svr_model.txt"
    elif mode == "speech":
        config.audio.sample_rate = 16000
        config.options.use_speech_scoring = True
        svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
    else:
        raise ValueError(f"Unrecognized mode: {mode}")
    config.options.svr_model_path = os.path.join(
        os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path)
    api = visqol_lib_py.VisqolApi()
    api.Create(config)
    similarity_result = api.Measure(ref_wav.astype(float), deg_wav.astype(float))
    return similarity_result.moslqo

def main():
    args = get_parser().parse_args()
    stoi_scores = []
    nb_pesq_scores = []
    wb_pesq_scores = []
    if not args.output_result_path.exists():
        args.output_result_path.mkdir(parents=True)
    deg_files = list(Path(args.deg_dir).rglob(f'*.{args.ext}'))
    with open(f"{args.output_result_path}/pesq_scores.txt","w") as p, open(f"{args.output_result_path}/stoi_scores.txt","w") as s:
        for idx, deg_wav_path in enumerate(tqdm(deg_files)):
            relative_path = deg_wav_path.relative_to(args.deg_dir)
            ref_wav_path = Path(args.ref_dir) / relative_path.parents[0] /deg_wav_path.name.replace(f'_bw{args.bandwidth}', '')
            # ref_wav_path = Path(args.ref_dir) / relative_path.parents[0] /deg_wav_path.name.replace(f'', '')
            try:
                ref_wav,_ = librosa.load(ref_wav_path, sr=args.sr)
                deg_wav,_ = librosa.load(deg_wav_path, sr=args.sr)
            except Exception as e:
                print(f"[LOAD-ERROR][{idx+1}/{len(deg_files)}] ref={ref_wav_path} deg={deg_wav_path}: {e}")
                if args.debug:
                    traceback.print_exc()
                continue

            if args.debug:
                print(f"[PAIR {idx+1}/{len(deg_files)}] ref={ref_wav_path} deg={deg_wav_path}")
                print(f"  sr={args.sr} len_ref={len(ref_wav)} len_deg={len(deg_wav)}")
                print(f"  ref_max_abs={np.max(np.abs(ref_wav)) if ref_wav.size else 0:.6f} deg_max_abs={np.max(np.abs(deg_wav)) if deg_wav.size else 0:.6f}")
                print(f"  ref_nan={int(np.isnan(ref_wav).sum()) if ref_wav.size else 0} deg_nan={int(np.isnan(deg_wav).sum()) if deg_wav.size else 0}")
            stoi_score = calculate_stoi(ref_wav, deg_wav, sr=args.sr)
            try:
                nb_pesq_score, wb_pesq_score, pesq_info = calculate_pesq(ref_wav, deg_wav, args.sr, debug=args.debug)
                if args.debug and pesq_info is not None:
                    print(f"  PESQ info: {pesq_info}")
            except Exception as e:
                print(f"[PESQ-ERROR][{idx+1}/{len(deg_files)}] {ref_wav_path} vs {deg_wav_path}: {e}")
                if args.debug:
                    traceback.print_exc()
                nb_pesq_score, wb_pesq_score = 0, 0
            nb_pesq_scores.append(nb_pesq_score)
            wb_pesq_scores.append(wb_pesq_score)
            p.write(f"{ref_wav_path}\t{deg_wav_path}\t{wb_pesq_score}\n")
            if stoi_score!=1e-5:
                stoi_scores.append(stoi_score)
                s.write(f"{ref_wav_path}\t{deg_wav_path}\t{stoi_score}\n")
    return np.mean(stoi_scores), np.mean(nb_pesq_scores), np.mean(wb_pesq_scores)
if __name__ == '__main__':
    mean_stoi, mean_nb_pesq, mean_wb_pesq = main()
    print(f"STOI: {mean_stoi}")
    print(f"NB PESQ: {mean_nb_pesq}")
    print(f"WB PESQ: {mean_wb_pesq}")
import mne
import numpy as np
import os
import pdb
from sklearn.decomposition import PCA

def select_and_save_channels(raw, save_path):
    raw.save(save_path, overwrite=True)
    print(f"已保存：{save_path}")

# def apply_pca_to_data(data, info, n_components=8):
#     """
#     对EEG数据应用PCA，降维到指定的通道数，并返回降维后的数据。
#     :param data: EEG数据，shape = (n_channels, n_times)
#     :param info: 原始数据的info对象，包含通道名称等信息
#     :param n_components: PCA的主成分个数，默认为10
#     :return: 降维后的数据，shape = (n_components, n_times)
#     """
#     # 转置数据，使得PCA应用在通道维度
#     data_reshaped = data.T  # 转置数据，shape = (n_times, n_channels)
    
#     # 应用PCA进行降维
#     pca = PCA(n_components=n_components)
#     pca_result = pca.fit_transform(data_reshaped)  # PCA降维
    
#     # 创建新的info对象
#     new_info = info.copy()
    
#     # 更新info中的通道名，使其适应降维后的通道数
#     new_info = mne.create_info(ch_names=[f"PC{i+1}" for i in range(n_components)], sfreq=info['sfreq'], ch_types="eeg")
    
    
#     # 返回降维后的数据和更新后的info
#     return pca_result.T, new_info
# 加载数据

# def apply_baseline(raw, baseline=(None, 0)):
#     """
#     对Raw对象进行基线校正
#     :param raw: mne.Raw对象
#     :param baseline: 基线时间范围，默认为(None, 0)
#     """
#     raw.apply_baseline(baseline)
#     print("基线校正完成")
#     return raw

def baseline_correction(data, raw, start_sample, baseline_duration=0.2, sfreq=500):
    """
    对 EEG 片段进行基线矫正
    :param data: (n_channels, n_times) 原始 EEG 片段数据
    :param raw: MNE Raw 对象
    :param start_sample: 事件起始采样点
    :param baseline_duration: 基线持续时间 (秒)
    :param sfreq: 采样率 (Hz)
    :return: 经过基线矫正后的数据
    """
    baseline_samples = int(baseline_duration * sfreq)  # 计算基线窗口大小
    baseline_start = max(0, start_sample - baseline_samples)  # 确保不会越界
    baseline_end = start_sample
    
    # 计算基线（每个通道的均值）
    baseline_mean = np.mean(raw[:, baseline_start:baseline_end][0], axis=1, keepdims=True)
    
    # 执行基线矫正
    corrected_data = data - baseline_mean  
    return corrected_data

def load_data(path = None):
    raw1 = mne.io.read_raw_brainvision(path, preload=True)
    # raw1 = remove_artifacts_ica()
    raw = raw1.copy()
    elc_file_path = '/root/code/preprocessing/standard_CA-209.elc'
    # montage = mne.channels.read_custom_montage(elc_file_path)
    # rotation_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    # rotation_matrix1 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    # positions = montage.get_positions()['ch_pos']
    # for electrode in positions.keys():
    #     cur = np.dot(rotation_matrix, positions[electrode])
    #     positions[electrode] = np.dot(rotation_matrix1, cur)
    # rotated_montage = mne.channels.make_dig_montage(ch_pos=positions, coord_frame='head')
    # raw.set_montage(rotated_montage)
    # fig = rotated_montage.plot()
    # fig.savefig('montage.png')

    # 获取事件信息
    events, _ = mne.events_from_annotations(raw)

    # 假设每个事件的ID从2到57
    start_event_mark = 2
    num_events = start_event_mark + 55
    events_per_group = 7
    groups = [list(range(i * events_per_group+start_event_mark, (i + 1) * events_per_group+start_event_mark)) for i in range(8)]  # 每组7个事件
    print(groups)
    event_ids = [event_id for event_id in range(2, num_events + 1)]  # 事件ID假设为2到57
    # 截取时间长度
    event_duration = 8  # 取每个事件后8秒的数据
    return raw, groups, event_ids, event_duration, events

def export_fif_dir_to_npy(in_dir: str,
                          out_dir: str,
                          expected_len_sec: float = 8.0,
                          sfreq: int = 500) -> None:
    """
    将目录下的 .fif 片段批量导出为 .npy，形状为 [channels, time]。

    参数
    ----
    in_dir: 包含 .fif 文件的输入目录
    out_dir: 输出 .npy 目录（会自动创建）
    expected_len_sec: 期望的片段时长（秒），默认 8 秒
    sfreq: 片段采样率（Hz），默认 500 Hz
    """
    os.makedirs(out_dir, exist_ok=True)
    expected_samples = int(expected_len_sec * sfreq)
    for fname in sorted(os.listdir(in_dir)):
        if not fname.lower().endswith('.fif'):
            continue
        fpath = os.path.join(in_dir, fname)
        raw = mne.io.read_raw_fif(fpath, preload=True, verbose=False)
        data = raw.get_data()  # (channels, time)
        # 对齐到固定长度
        if data.shape[1] < expected_samples:
            pad = expected_samples - data.shape[1]
            data = np.pad(data, ((0, 0), (0, pad)), mode='constant')
        elif data.shape[1] > expected_samples:
            data = data[:, :expected_samples]
        data = data.astype(np.float32, copy=False)
        out_name = os.path.splitext(fname)[0] + '.npy'
        np.save(os.path.join(out_dir, out_name), data)

def main():
    # 遍历每组，取前5个事件并截取10秒
    save_root = '/root/autodl-tmp/eeg_processed'
    # os.mkdir(save_root)
    for k, _, i in os.walk('/root/autodl-tmp/eeg_raw/train'):
        result = [element for element in i if 'vhdr' in element]
        if len(result) != 0:
            path = k+'/'+result[0]
            print(path)
            name = result[0].split('_')[0] + result[0].split('_')[1]
        else:
            continue

        raw, groups, event_ids1, duration, events=load_data(path)
        for group_idx, event_ids in enumerate(groups):
            save_dir = os.path.join(save_root, str(group_idx))
            os.makedirs(save_dir, exist_ok=True)
            
            # 每组取前5个事件
            for order_idx in range(5):
                target_id = event_ids[order_idx]
                print(target_id)
                # 查找匹配事件
                hz = events[target_id][0]
                print(hz)
                
                start_sample = hz
                print(start_sample)
                start_time = start_sample / 500.0
                end_time = start_time + duration
                
                # 边界检查
                if end_time > raw.times[-1]:
                    print(f"事件{target_id}超出数据范围，跳过")
                    continue
                
                data, times = raw[:, start_sample:start_sample+4000]  # 截取指定采样点的数据
                corrected_data = baseline_correction(data, raw, start_sample)
                # channels_to_keep = ["M2", "T8", "CP6", "P8", "FC6", "POz", "FC5", "T7", "P7", "CP5"]
                # info = mne.create_info(ch_names=[f"EEG{i+1}" for i in range(32)], sfreq=500, ch_types="eeg")
                # pca_corrected_data, new_info= apply_pca_to_data(corrected_data, info)
                
                # segment = mne.io.RawArray(pca_corrected_data, new_info)
                
                info = raw.info.copy()
                segment = mne.io.RawArray(corrected_data, info)

                select_and_save_channels(segment, os.path.join(save_dir, f"{name}_event{order_idx + 1}_id{target_id}.fif"))
        

if __name__ == '__main__':
    main()
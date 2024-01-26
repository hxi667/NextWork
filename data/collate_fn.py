import math
import random
import numpy as np
from utils.msg_manager import get_msg_mgr


class CollateFn(object):
    def __init__(self, label_set, sample_config):
        self.label_set = label_set
        sample_type = sample_config['sample_type']
        sample_type = sample_type.split('_')
        self.sampler = sample_type[0] # all | fixed | unfixed
        self.ordered = sample_type[1] # unordered | ordered
        if self.sampler not in ['fixed', 'unfixed', 'all']:
            raise ValueError
        if self.ordered not in ['ordered', 'unordered']:
            raise ValueError
        self.ordered = sample_type[1] == 'ordered'

        # 采用固定的帧数进行采样
        if self.sampler == 'fixed':
            self.frames_num_fixed = sample_config['frames_num_fixed'] # 帧数固定时进行训练的帧数

        # 采用不固定的帧数进行采样
        if self.sampler == 'unfixed':
            self.frames_num_max = sample_config['frames_num_max'] # 帧数非固定时进行训练的最大帧数
            self.frames_num_min = sample_config['frames_num_min'] # 帧数非固定时进行训练的最小帧数

        # 不使用全部帧, 并且帧有序的进行采样
        if self.sampler != 'all' and self.ordered:
            self.frames_skip_num = sample_config['frames_skip_num'] # 从采样序列中随机跳过的帧数

        # 使用全部帧进行采样时，如果配置文件中有 'frames_all_limit' 参数，则限制采样的帧数，防止内存不足
        self.frames_all_limit = -1 
        if self.sampler == 'all' and 'frames_all_limit' in sample_config:
            self.frames_all_limit = sample_config['frames_all_limit']

    def __call__(self, batch):
        # 入参 batch 表示一个 batch 大小的 data
        # len(batch) -> batch_size[0] * batch_size[1] / gpus num
        # len(batch[0]) -> 2
        # batch[0][0] -> [(133, 3, 128, 128), (133, 128, 128), ...] , 表示 data 
        # batch[0][1] -> ['001', 'bg-01', '000', ['./001/bg-01/000/000-rgbs.pkl', './001/bg-01/000/000-sils.pkl', ...]] , 表示 data info 
        batch_size = len(batch)
        ### currently, the functionality of feature_num is not fully supported yet, it refers to 1 now. We are supposed to make our framework support multiple source of input data, such as silhouette, or skeleton.
        # 目前 gaitedge 支持多种数据源
        feature_num = len(batch[0][0]) # 2 # 多种数据源
        seqs_batch, labs_batch, typs_batch, vies_batch = [], [], [], []

        for bt in batch:
            seqs_batch.append(bt[0]) # [[(133, 3, 128, 128), (133, 128, 128), ...], ...]
            labs_batch.append(self.label_set.index(bt[1][0])) # [['001'], ...]
            typs_batch.append(bt[1][1]) # [['bg-01'], ...]
            vies_batch.append(bt[1][2]) # [['000'], ...]

        global count
        count = 0

        def sample_frames(seqs):
            # seqs -> [(133, 3, 128, 128), (133, 128, 128), ...]
            global count
            sampled_fras = [[] for i in range(feature_num)]
            seq_len = len(seqs[0]) # 133
            indices = list(range(seq_len)) # [0, 1, 2, ..., 130, 131, 132]
            # 如果 self.sampler 为 "all", NoOp
            if self.sampler in ['fixed', 'unfixed']:
                # 采样的帧数是固定的
                if self.sampler == 'fixed':
                    frames_num = self.frames_num_fixed
                # 采样的帧数是不固定的
                else:
                    frames_num = random.choice(
                        list(range(self.frames_num_min, self.frames_num_max+1)))  # 从 [frames_num_min, frames_num_max+1] 区间选择一个随机数作为采样的帧数
                
                # 有序采样
                if self.ordered:
                    fs_n = frames_num + self.frames_skip_num
                    # 如果数据序列的长度小于 fs_n ，则重复数据以满足要求的帧数 frames_num
                    if seq_len < fs_n:
                        it = math.ceil(fs_n / seq_len)
                        seq_len = seq_len * it
                        indices = indices * it

                    start = random.choice(list(range(0, seq_len - fs_n + 1))) # 从 [0, seq_len - fs_n + 1] 区间内随机选择起始帧
                    end = start + fs_n
                    idx_lst = list(range(seq_len))
                    # 例: [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74]
                    idx_lst = idx_lst[start:end] 
                    # eg: 如果 frames_skip_num 变量取值为 4 -> 那么采样时随机跳过 4 帧-> 跳过 [43, 53, 54, 61] ->
                    # [41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 55, 56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74]
                    idx_lst = sorted(np.random.choice(
                        idx_lst, frames_num, replace=False)) # 从 "idx_lst" 中随机选择 "frames_num" 数量的帧，然后排序
                    indices = [indices[i] for i in idx_lst]
                # 无序采样
                else:
                    replace = seq_len < frames_num # boolean ， 根据序列的长度判断是否需要使用替换（replace）进行抽样

                    if seq_len == 0:
                        get_msg_mgr().log_debug('Find no frames in the sequence %s-%s-%s.'
                                                % (str(labs_batch[count]), str(typs_batch[count]), str(vies_batch[count])))

                    count += 1   
                    # 例: array([74, 64, 64, 52, 67, 65, 44, 74, 45, 55, 66, 74, 48, 46, 73, 62, 69, 71, 57, 41, 45, 52, 59, 69, 42, 68, 48, 57, 74, 49])
                    indices = np.random.choice(
                        indices, frames_num, replace=replace) # 随机排序
            
            # 遍历多种数据源
            for i in range(feature_num):
                for j in indices[:self.frames_all_limit] if self.frames_all_limit > -1 and len(indices) > self.frames_all_limit else indices:
                    sampled_fras[i].append(seqs[i][j])
            return sampled_fras

        # f: feature_num
        # b: batch_size
        # p: batch_size_per_gpu
        # g: gpus_num
        fras_batch = [sample_frames(seqs) for seqs in seqs_batch]  # fras_batch.size == [b, f]
        batch = [fras_batch, labs_batch, typs_batch, vies_batch, None]

        # 采样的帧数是固定的
        if self.sampler == "fixed":
            # 将 fras_batch 中的每个元素转换为 NumPy 数组，得到一个新的 fras_batch。这个新的 fras_batch 是一个包含多个特征和 batch 的列表，维度为 [feature_num, batch_size]
            fras_batch = [[np.asarray(fras_batch[i][j]) for i in range(batch_size)]
                          for j in range(feature_num)]  # fras_batch.size == [f, b]??
        # 采样的帧数是不固定的
        else:
            seqL_batch = [[len(fras_batch[i][0])
                           for i in range(batch_size)]]  # seqL_batch.size == [1, p]??

            def my_cat(k): return np.concatenate(
                [fras_batch[i][k] for i in range(batch_size)], 0) # "0" means "axis=0"
            
            fras_batch = [[my_cat(k)] for k in range(feature_num)]  # fras_batch.size == [f, g] ??

            batch[-1] = np.asarray(seqL_batch)

        # 更新 batch[0]
        batch[0] = fras_batch
        # batch[0] -> [[No.1(seqL_batch, 3, 128, 128), No.2(seqL_batch, 3, 128, 128), ..., No.batchsize(seqL_batch, 3, 128, 128)], [No.1(seqL_batch, 128, 128), No.2(seqL_batch, 128, 128), ..., No.batchsize(seqL_batch, 128, 128)], ...]
        # batch[1] -> [1, ...]
        # batch[2] -> ['bg-01', ...]
        # batch[3] -> ['000', ...]
        # batch[4] ->->  如果为fixed -> 则为None, 如果为 unfixed -> np.asarray: seq length of each batch 
        return batch

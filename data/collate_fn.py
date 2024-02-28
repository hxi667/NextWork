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

        # Sampling with a fixed number of frames
        if self.sampler == 'fixed':
            self.frames_num_fixed = sample_config['frames_num_fixed'] # Number of frames for training when the number of frames is fixed

        # Sampling with a unfixed number of frames
        if self.sampler == 'unfixed':
            self.frames_num_max = sample_config['frames_num_max'] # Maximum number of frames for training when the number of frames is not fixed
            self.frames_num_min = sample_config['frames_num_min'] # Minimum number of frames for training when the number of frames is not fixed

        # Not all frames are used, and frames are sampled sequentially.
        if self.sampler != 'all' and self.ordered:
            self.frames_skip_num = sample_config['frames_skip_num'] # Number of frames randomly skipped from the sampling sequence

        # When sampling with all frames, if there is a 'frames_all_limit' parameter in the config file, the number of frames sampled will be limited to prevent running out of memory.
        self.frames_all_limit = -1 
        if self.sampler == 'all' and 'frames_all_limit' in sample_config:
            self.frames_all_limit = sample_config['frames_all_limit']

    def __call__(self, batch):
        # The input parameter "batch" represents a batch size of data.
        # len(batch) -> batch_size[0] * batch_size[1] / gpus num
        # len(batch[0]) -> 2
        # batch[0][0] -> [(133, 3, 128, 128), (133, 128, 128), ...], represent data 
        # batch[0][1] -> ['001', 'bg-01', '000', ['./001/bg-01/000/000-rgbs.pkl', './001/bg-01/000/000-sils.pkl', ...]], represent data info 
        batch_size = len(batch)
        ### currently, the functionality of feature_num is not fully supported yet, it refers to 1 now. We are supposed to make our framework support multiple source of input data, such as silhouette, or skeleton.
        # Currently gaitedge supports multiple data sources
        feature_num = len(batch[0][0]) # 2 # multiple data sources
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
            # if self.sampler is "all", NoOp
            if self.sampler in ['fixed', 'unfixed']:
                # Sampling with a fixed number of frames
                if self.sampler == 'fixed':
                    frames_num = self.frames_num_fixed
                # Sampling with a unfixed number of frames
                else:
                    # Select a random number from the interval [frames_num_min, frames_num_max+1] as the number of frames to be sampled
                    frames_num = random.choice(
                        list(range(self.frames_num_min, self.frames_num_max+1))) 
                
                # ordered sampling
                if self.ordered:
                    fs_n = frames_num + self.frames_skip_num
                    # If the length of the data sequence is less than fs_n, repeat the data to meet the required number of frames frames_num
                    if seq_len < fs_n:
                        it = math.ceil(fs_n / seq_len)
                        seq_len = seq_len * it
                        indices = indices * it

                    start = random.choice(list(range(0, seq_len - fs_n + 1))) # Randomly select the start frame from the interval [0, seq_len - fs_n + 1].
                    end = start + fs_n
                    idx_lst = list(range(seq_len))
                    # eg: [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74]
                    idx_lst = idx_lst[start:end] 
                    # eg:If the "frames_skip_num" takes the value 4 -> Randomly skip 4 frames during sampling -> skip [43, 53, 54, 61] ->
                    # [41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 55, 56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74]
                    idx_lst = sorted(np.random.choice(
                        idx_lst, frames_num, replace=False)) # Select a random number of frames from "idx_lst" with the number of "frames_num" and sort them.
                    indices = [indices[i] for i in idx_lst]
                # unordered sampling
                else:
                    replace = seq_len < frames_num # boolean, determine if sampling with "replace" is required based on the length of the sequence

                    if seq_len == 0:
                        get_msg_mgr().log_debug('Find no frames in the sequence %s-%s-%s.'
                                                % (str(labs_batch[count]), str(typs_batch[count]), str(vies_batch[count])))

                    count += 1   
                    # eg: array([74, 64, 64, 52, 67, 65, 44, 74, 45, 55, 66, 74, 48, 46, 73, 62, 69, 71, 57, 41, 45, 52, 59, 69, 42, 68, 48, 57, 74, 49])
                    indices = np.random.choice(
                        indices, frames_num, replace=replace) # random order
            
            # Iterate over multiple data sources
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

        # The number of frames sampled is fixed
        if self.sampler == "fixed":
            # Convert each element of "fras_batch" to a NumPy array to get a new "fras_batch".
            # This new "fras_batch" is a list of multiple features and batch with dimensions [feature_num, batch_size]
            fras_batch = [[np.asarray(fras_batch[i][j]) for i in range(batch_size)]
                          for j in range(feature_num)]  # fras_batch.size == [f, b]??
        # The number of frames sampled is unfixed
        else:
            seqL_batch = [[len(fras_batch[i][0])
                           for i in range(batch_size)]]  # seqL_batch.size == [1, p]??

            def my_cat(k): return np.concatenate(
                [fras_batch[i][k] for i in range(batch_size)], 0) # "0" means "axis=0"
            
            fras_batch = [[my_cat(k)] for k in range(feature_num)]  # fras_batch.size == [f, g] ??

            batch[-1] = np.asarray(seqL_batch)

        # update batch[0]
        batch[0] = fras_batch
        # batch[0] -> [[No.1(seqL_batch, 3, 128, 128), No.2(seqL_batch, 3, 128, 128), ..., No.batchsize(seqL_batch, 3, 128, 128)], [No.1(seqL_batch, 128, 128), No.2(seqL_batch, 128, 128), ..., No.batchsize(seqL_batch, 128, 128)], ...]
        # batch[1] -> [1, ...]
        # batch[2] -> ['bg-01', ...]
        # batch[3] -> ['000', ...]
        # batch[4] ->->  if fixed -> None, if unfixed -> np.asarray: seq length of each batch 
        return batch

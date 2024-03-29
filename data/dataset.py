import os
import pickle
import os.path as osp
import torch.utils.data as tordata
import json
from utils.msg_manager import get_msg_mgr


class DataSet(tordata.Dataset):
    def __init__(self, data_cfg, training):
        """
            seqs_info: the list with each element indicating 
                            a certain gait sequence presented as [label, type, view, paths];


            return getitem[idx]: data_list, seq_info
                                    data_list.shape -> [(133, 3, 128, 128), (133, 128, 128), ...]
                                    seq_info -> ['001', 'bg-01', '000', ['./001/bg-01/000/000-rgbs.pkl', './001/bg-01/000/000-sils.pkl', ...]]
        """

        # get self.seqs_info variable -> [['001', 'bg-01', '000', ['data_path/001/bg-01/000/000-rgbs.pkl', 'data_path/001/bg-01/000/000-sils.pkl', ...]], ...] 
        self.__dataset_parser(data_cfg, training) 
        self.cache = data_cfg['cache']
        
        self.label_list = [seq_info[0] for seq_info in self.seqs_info]
        self.types_list = [seq_info[1] for seq_info in self.seqs_info]
        self.views_list = [seq_info[2] for seq_info in self.seqs_info]

        self.label_set = sorted(list(set(self.label_list))) # the list was sorted and labels unique.
        self.types_set = sorted(list(set(self.types_list)))
        self.views_set = sorted(list(set(self.views_list)))
        
        # save seqs data
        self.seqs_data = [None] * len(self) 
        self.indices_dict = {label: [] for label in self.label_set}
        
        # self.indices_dict -> {'001': [0, 1, 2, ... , 107, 108, 109], '002': [110, 111, 112, ... , 217, 218, 219], ...} 
        for i, seq_info in enumerate(self.seqs_info):
            self.indices_dict[seq_info[0]].append(i) # eg: seq_info[0] -> "001", the key of self.indices_dict
        
        # if true, load all data to memory
        if self.cache:
            self.__load_all_data()

    def __len__(self):
        return len(self.seqs_info)

    def __loader__(self, paths):
        # type(paths): list
        # paths -> ['./001/bg-01/000/000-rgbs.pkl', './001/bg-01/000/000-sils.pkl', ...]
        paths = sorted(paths)
        data_list = []
        for pth in paths:
            if pth.endswith('.pkl'):
                with open(pth, 'rb') as f:
                    _ = pickle.load(f)
                f.close()
            else:
                raise ValueError('- Loader - just support .pkl !!!')
            data_list.append(_)
        
        for idx, data in enumerate(data_list):
            if len(data) != len(data_list[0]):
                raise ValueError(
                    'Each input data({}) should have the same length.'.format(paths[idx]))
            if len(data) == 0:
                raise ValueError(
                    'Each input data({}) should have at least one element.'.format(paths[idx]))
        return data_list

    def __getitem__(self, idx):
        # if self.cache is false, when training, load data and return the data_list and seq_info by idx index.
        if not self.cache:
            data_list = self.__loader__(self.seqs_info[idx][-1])
        # if self.cache is true，load all data before training, and save data and info in the memory variables self.seqs_data and self.seqs_info
        elif self.seqs_data[idx] is None:
            data_list = self.__loader__(self.seqs_info[idx][-1])
            self.seqs_data[idx] = data_list
        else:
            data_list = self.seqs_data[idx]
        seq_info = self.seqs_info[idx]
        # eg: data_list.shape -> [(133, 3, 128, 128), (133, 128, 128), ...]
        # eg: seq_info -> ['001', 'bg-01', '000', ['./001/bg-01/000/000-rgbs.pkl', './001/bg-01/000/000-sils.pkl', ...]]
        return data_list, seq_info

    def __load_all_data(self):  
        for idx in range(len(self)):
            self.__getitem__(idx)

    # get "self.seqs_info" 变量, return a list -> [['001', 'bg-01', '000', ['./001/bg-01/000/000-rgbs.pkl', './001/bg-01/000/000-sils.pkl', ...]], ...] 
    def __dataset_parser(self, data_config, training):
        dataset_root = data_config['dataset_root']
        try:
            data_in_use = data_config['data_in_use']  #  True or Fals. For multiple data types: [True | False, True | False, ..., True | False], eg：[True, False, False, False]
        except:
            data_in_use = None

        with open(data_config['dataset_partition'], "rb") as f:
            partition = json.load(f)
        train_set = partition["TRAIN_SET"]
        test_set = partition["TEST_SET"]
        label_list = os.listdir(dataset_root)
        # Iterate "train_set" variable, get the current element "label", if "label" is in "label_list", then save "label" in the new list.
        train_set = [label for label in train_set if label in label_list]
        test_set = [label for label in test_set if label in label_list]
        miss_pids = [label for label in label_list if label not in (
            train_set + test_set)]
        
        msg_mgr = get_msg_mgr()
        # write pid list info to console/ logs file
        def log_pid_list(pid_list):
            if len(pid_list) >= 3:
                msg_mgr.log_info('[%s, %s, ..., %s]' %
                                 (pid_list[0], pid_list[1], pid_list[-1]))             
            else:
                msg_mgr.log_info(pid_list)
                

        if len(miss_pids) > 0:
            msg_mgr.log_debug('Miss Pid List: %s' % miss_pids)
            msg_mgr.log_debug(miss_pids)

        if training:
            msg_mgr.log_info('Train Pid List: ')
            log_pid_list(train_set)
        else:
            msg_mgr.log_info("Test Pid List: ")
            log_pid_list(test_set)


        #  return seqs_info_list. eg: seqs_info_list[0] = ['001', 'bg-01', '000', ['./001/bg-01/000/000-rgbs.pkl', './001/bg-01/000/000-sils.pkl', '...']] 
        def get_seqs_info_list(label_set):
            seqs_info_list = [] 
            for lab in label_set:
                for typ in sorted(os.listdir(osp.join(dataset_root, lab))):
                    for vie in sorted(os.listdir(osp.join(dataset_root, lab, typ))):
                        seq_info = [lab, typ, vie]
                        seq_path = osp.join(dataset_root, *seq_info)
                        seq_dirs = sorted(os.listdir(seq_path)) #
                        if seq_dirs != []: # ['000-rgbs.pkl', '000-sils.pkl', '...']
                            # seqs path
                            seq_dirs = [osp.join(seq_path, dir)
                                        for dir in seq_dirs]
                            # 去掉不需要的数据类型
                            if data_in_use is not None:
                                seq_dirs = [dir for dir, use_bl in zip(
                                    seq_dirs, data_in_use) if use_bl]
                            seqs_info_list.append([*seq_info, seq_dirs])
                        else:
                            msg_mgr.log_debug(
                                'Find no .pkl file in %s-%s-%s.' % (lab, typ, vie))
            return seqs_info_list

        self.seqs_info = get_seqs_info_list(
            train_set) if training else get_seqs_info_list(test_set)

import time
import torch

import numpy as np
import torchvision.utils as vutils
import os.path as osp
from time import strftime, localtime

from torch.utils.tensorboard import SummaryWriter
from .common import is_list, is_tensor, ts2np, mkdir, Odict, NoOp
import logging


# The class of setting SummaryWriter and logger, and output logs to tensorboard, console and  logs file 
class MessageManager:
    def __init__(self):
        self.info_dict = Odict()
        self.writer_hparams = ['image', 'scalar'] # parameters of tensorboard
        self.time = time.time()
    
    # init SummaryWriter and logger
    def init_manager(self, save_path, log_to_file, log_iter, iteration=0):
        self.iteration = iteration
        self.log_iter = log_iter
        mkdir(osp.join(save_path, "summary/"))
        # init SummaryWriter, output logs to tensorboard
        self.writer = SummaryWriter(
            osp.join(save_path, "summary/"), purge_step=self.iteration)
        # init logger, output logs to console and logs file
        self.init_logger(save_path, log_to_file)

    # init logger, output logs to console and logs file
    def init_logger(self, save_path, log_to_file):
        # init logger
        self.logger = logging.getLogger('Gait_Gait')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        if log_to_file:
            mkdir(osp.join(save_path, "logs/"))
            vlog = logging.FileHandler(
                osp.join(save_path, "logs/", strftime('%Y-%m-%d-%H-%M-%S', localtime())+'.txt'))
            vlog.setLevel(logging.INFO)
            vlog.setFormatter(formatter)
            self.logger.addHandler(vlog)

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(logging.DEBUG)
        self.logger.addHandler(console)

    # append info to info_dict
    def append(self, info):
        for k, v in info.items():
            v = [v] if not is_list(v) else v
            v = [ts2np(_) if is_tensor(_) else _ for _ in v]
            info[k] = v
        self.info_dict.append(info)

    # refresh info_dict to disk
    def flush(self):
        self.info_dict.clear()
        self.writer.flush()

    # write summary to tensorboard
    def write_to_tensorboard(self, summary):

        for k, v in summary.items():
            module_name = k.split('/')[0]
            if module_name not in self.writer_hparams:
                self.log_warning(
                    'Not Expected --Summary-- type [{}] appear!!!{}'.format(k, self.writer_hparams))
                continue
            board_name = k.replace(module_name + "/", '')
            writer_module = getattr(self.writer, 'add_' + module_name)
            v = v.detach() if is_tensor(v) else v
            v = vutils.make_grid(
                v, normalize=True, scale_each=True) if 'image' in module_name else v
            if module_name == 'scalar':
                try:
                    v = v.mean()
                except:
                    v = v
            writer_module(board_name, v, self.iteration)

    # write training info to log
    def log_training_info(self):
        now = time.time()
        string = "Iteration {:0>5}, Cost {:.2f}s".format(
            self.iteration, now-self.time, end="")
        for i, (k, v) in enumerate(self.info_dict.items()):
            if 'scalar' not in k:
                continue
            k = k.replace('scalar/', '').replace('/', '_')
            end = "\n" if i == len(self.info_dict)-1 else ""
            string += ", {0}={1:.4f}".format(k, np.mean(v), end=end)
        self.log_info(string)
        self.reset_time()

    # reset time to current time 
    def reset_time(self):
        self.time = time.time()

    # When training, write "loss_info" and "visual_summary" info at every step to log and tensorboard 
    def train_step(self, info, summary):
        self.iteration += 1
        self.append(info)
        if self.iteration % self.log_iter == 0:
            self.log_training_info()
            self.flush()
            self.write_to_tensorboard(summary)

    # write DEBUG level info to console/logs file
    def log_debug(self, *args, **kwargs):
        self.logger.debug(*args, **kwargs)

    # write INFO level info to console/logs file
    def log_info(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)

    # write WARNING level info to console/logs file
    def log_warning(self, *args, **kwargs):
        self.logger.warning(*args, **kwargs)


# instance MessageManager class
msg_mgr = MessageManager()
noop = NoOp()


# return MessageManager instance
def get_msg_mgr():
    if torch.distributed.get_rank() > 0:
        return noop
    else:
        return msg_mgr

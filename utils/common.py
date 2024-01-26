import torch.nn as nn
from collections import OrderedDict
import inspect
import os
import sys
import logging
import torch
import random
import time
import numpy as np
import copy
import torch.autograd as autograd
from torch.nn.parallel import DistributedDataParallel as DDP


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


# NoOp 是一个空类，它的目的是在访问不存在的属性时返回一个无操作的函数
class NoOp:
    def __getattr__(self, *args):
        def no_op(*args, **kwargs): pass
        return no_op


# dictionary object
class Odict(OrderedDict):
    def append(self, odict):
        dst_keys = self.keys()
        for k, v in odict.items():
            if not is_list(v):
                v = [v]
            if k in dst_keys:
                if is_list(self[k]):
                    self[k] += v
                else:
                    self[k] = [self[k]] + v
            else:
                self[k] = v


# 验证 obj 的参数
def get_valid_args(obj, input_args, free_keys=[]):
    '''
        这个函数主要用于从输入参数 input_args 中提取那些在 obj 函数中的形参或 obj 类的 __init__ 方法中被预期的参数。
        对于那些不在预期参数列表中的键，会记录日志提示，但不会导致函数出错。
        这样的设计在处理配置或者选项时比较灵活，允许输入参数中包含一些额外的信息。
    '''
    if inspect.isfunction(obj):
        # inspect.getfullargspec(obj)[0]: 获取 obj 函数的参数名称列表
        expected_keys = inspect.getfullargspec(obj)[0]
    elif inspect.isclass(obj):
        # inspect.getfullargspec(obj.__init__)[0]: 获取 obj 类初始化方法 (__init__) 中的参数名列表
        expected_keys = inspect.getfullargspec(obj.__init__)[0]
    else:
        raise ValueError('Just support function and class object!')
    unexpect_keys = list()
    expected_args = {}
    for k, v in input_args.items():
        if k in expected_keys:
            expected_args[k] = v 
        # 如果 key 不在 expected_keys 中，但是在 "free_keys" 中，直接跳过不处理
        elif k in free_keys:
            pass
        else:
            # 如果 key 既不在 expected_keys 中也不在 free_keys 中，将该 key 添加到 unexpect_keys 列表中，并记录日志
            unexpect_keys.append(k)
    if unexpect_keys != []:
        logging.info("Find Unexpected Args(%s) in the Configuration of - %s -" %
                     (', '.join(unexpect_keys), obj.__name__))
    return expected_args


# 从 sources（一个或多个对象） 中获取名为 name 的属性 
def get_attr_from(sources, name):
    try:
        return getattr(sources[0], name)
    except:
        return get_attr_from(sources[1:], name) if len(sources) > 1 else getattr(sources[0], name)

# 如果 x 是 list 或者 tuple 类型, 返回 true
def is_list_or_tuple(x):
    return isinstance(x, (list, tuple))

# 如果 x 是 list 或者 nn.ModuleList 类型, 返回 true
def is_list(x):
    return isinstance(x, list) or isinstance(x, nn.ModuleList)


# 如果 x 是 dict ，OrderedDict 或 Odict 类型, 返回 true
def is_dict(x):
    return isinstance(x, dict) or isinstance(x, OrderedDict) or isinstance(x, Odict)


# 如果 x 是 torch.Tensor 类型, 返回 true
def is_tensor(x):
    return isinstance(x, torch.Tensor)


# tensor2numpy
def ts2np(x):
    return x.cpu().data.numpy()


# tensor2variable 
def ts2var(x, **kwargs):
    return autograd.Variable(x, **kwargs).cuda()

# numpy2variable
def np2var(x, **kwargs):
    return ts2var(torch.from_numpy(x), **kwargs)


# list2variable
def list2var(x, **kwargs):
    return np2var(np.array(x), **kwargs)


# mkdir
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# clone the module N times to form the nn.ModuleList
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 初始化随机种子
def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# 分布式
def ddp_all_gather(features, dim=0, requires_grad=True):
    '''
        在分布式训练环境中实现 All-Gather 操作，确保每个设备上的模型都有完整的数据集信息
        inputs: [n, ...]
    '''

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    feature_list = [torch.ones_like(features) for _ in range(world_size)]
    torch.distributed.all_gather(feature_list, features.contiguous())

    if requires_grad:
        feature_list[rank] = features
    feature = torch.cat(feature_list, dim=dim)
    return feature


# https://github.com/pytorch/pytorch/issues/16885
class DDPPassthrough(DDP):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


# 返回一个经过分布式数据并行 (DDP) 化处理的 module
def get_ddp_module(module, find_unused_parameters=False, **kwargs):
    # 检查输入 module 是否包含参数（权重和偏置等），如果没有，则直接返回原始 module
    if len(list(module.parameters())) == 0: 
        return module
    device = torch.cuda.current_device()
    module = DDPPassthrough(module, device_ids=[device], output_device=device,
                            find_unused_parameters=find_unused_parameters, **kwargs)
    return module

# 模型参数计数
def params_count(net):
    n_parameters = sum(p.numel() for p in net.parameters())
    if hasattr(net, '__name__'):
        return 'Parameters Count ==> {}: {:.5f}M'.format(net.__name__, n_parameters / 1e6)
    else:
        return 'Parameters Count ==> {:.5f}M'.format(n_parameters / 1e6)

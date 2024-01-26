from data.dataset import DataSet
from utils.common import get_valid_args, get_attr_from
import torch.utils.data as tordata
import data.sampler as Samplers
from data.collate_fn import CollateFn

# 获取 data loader
def get_loader(cfgs, train=True):
    sampler_cfg = cfgs['trainer_cfg']['sampler'] if train else cfgs['evaluator_cfg']['sampler']
    # DataSet
    dataset = DataSet(cfgs['data_cfg'], train)
    
    # 从 Samplers 包中获取名为 sampler_cfg['type'] 的 Samplers 类 
    Sampler = get_attr_from([Samplers], sampler_cfg['type'])
    vaild_args = get_valid_args(Sampler, sampler_cfg, free_keys=[
        'sample_type', 'type'])
    # 实例化这个 Sampler 类
    sampler = Sampler(dataset, **vaild_args)

    loader = tordata.DataLoader(
        dataset=dataset,
        batch_sampler=sampler, # batch_sampler: 每次返回一个批次的索引
        collate_fn=CollateFn(dataset.label_set, sampler_cfg),
        num_workers=cfgs['data_cfg']['num_workers'])
    return loader


import torch

from models import *
import models.gaitgl as gaitgl
import models.gaitset as gaitset
import models.gaitpart as gaitpart
import models.baseline as baseline

from utils import get_msg_mgr


# load checkpoint
def load_ckpt(device, model, save_name):
    
    msg_mgr = get_msg_mgr()

    checkpoint = torch.load(save_name, map_location=torch.device("cuda", device))
    model_state_dict = checkpoint['model'] # checkpoint['optimizer'], checkpoint['scheduler']

    # 如果 not load_ckpt_strict 为 True，即不是严格检查checkpoint是否与定义的模型相同，
    # 则找到两个模型状态字典中共有的参数键，并将它们按照某个规则排序后打印输出
    if not model.load_ckpt_strict:
        msg_mgr.log_info("-------- Restored Params List --------")
        msg_mgr.log_info(sorted(set(model_state_dict.keys()).intersection(
            set(model.state_dict().keys()))))
    
    # 参数strict默认是True，这时候就需要严格按照模型中参数的Key值来加载参数，如果增删了模型的结构层，或者改变了原始层中的参数，加载就会报错
    # 参数strict如果为Flase，就可以只加载具有相同名称的参数层，对于修改的模型结构层进行随机赋值。
    # 这里需要注意的是，如果只是改变了原来层的参数，但是没有换名称，依然还是会报错。因为根据key值找到对应的层之后，进行赋值，发现参数不匹配。
    # 这时候可以将原来的层换个名称，再加载就不会报错了。最后，大家需要注意的是，strict=Flase要谨慎使用，因为很有可能你会一点参数也没加载进来
    model.load_state_dict(model_state_dict, strict=model.load_ckpt_strict)
    
    msg_mgr.log_info("Restore Parameters from %s !!!" % save_name)


# load model from checkpoint
def resume_ckpt(device, model, dataset_name):
    # if type(restore_hint) == int
    if isinstance(model.restore_hint, int):
        save_name = './checkpoint/{}/{}/{}-{:0>5}.pt'.format(dataset_name, model.__name__, model.__name__, model.restore_hint)
    # if type(restore_hint) == string
    elif isinstance(model.restore_hint, str):
        save_name = model.restore_hint
    else:
        raise ValueError(
            "Error type for -Restore_Hint-, supported: int or string.")
    load_ckpt(device, model, save_name)



def get_teachers_student(model_cfg, dataset_name):
    # Cuda setup
    device = torch.distributed.get_rank()
    torch.cuda.set_device(device)

    model_map = {"GaitSet": gaitset.gaitSet,
                 "GaitPart": gaitpart.gaitPart,
                 "GaitGL": gaitgl.gaitGL,
                 'Baseline_ResNet9': baseline.baseline_ResNet9}

    # Teachers setup
    teachers = []
    # Add teachers models into teacher model list
    for t in model_cfg["teachers"]:
        if t['name'] in model_map:
            net = model_map[t['name']]()  # eg: GaitSet()
            net.__name__ = t['name']
            net.restore_hint = t['restore_hint']
            net.load_ckpt_strict = t['load_ckpt_strict']
            teachers.append(net)

    assert len(teachers) > 0, "Teachers must be in %s" % " ".join(model_map.keys)

    for i, teacher in enumerate(teachers):
        for p in teacher.parameters():
            p.requires_grad = False
        teacher = teacher.to(device=torch.device("cuda", device))
        teachers[i].__name__ = teacher.__name__
        teachers[i].restore_hint = teacher.restore_hint
        teachers[i].load_ckpt_strict = teachers[i].load_ckpt_strict

    # Load parameters in teacher models
    for teacher in teachers:
        resume_ckpt(device, teacher, dataset_name)
 
    if model_cfg["teacher_eval"]:
        for teacher in teachers:
            teacher.eval()

    # Student setup
    assert model_cfg["student"] in model_map, "Student must be in %s" % " ".join(model_map.keys)
    student = model_map[model_cfg["student"]]() # eg: GaitSet()
    student.__name__ = model_cfg["student"]
    student = student.to(device=torch.device("cuda", device))
    #######################
    if device == "cuda":
        out_dims = student.out_dims
        student = torch.nn.DataParallel(student)
        student.out_dims = out_dims
    #######################
    return teachers, student
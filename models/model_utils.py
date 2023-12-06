import torch

from models import *
import models.gaitgl as gaitgl
import models.gaitset as gaitset
import models.gaitpart as gaitpart
import models.baseline as baseline

from utils import get_msg_mgr

def get_teachers_student(model_cfg):
    teachers = []

    model_map = {"GaitSet": gaitset.gaitSet,
                 "GaitPart": gaitpart.gaitPart,
                 "GaitGL": gaitgl.gaitGL,
                 'Baseline_ResNet9': baseline.baseline_ResNet9}

    # Add teachers models into teacher model list
    for t in model_cfg["teachers"]:
        if t['name'] in model_map:
            net = model_map[t['name']]()  # eg: GaitSet()
            net.__name__ = t['name']
            teachers.append(net)

    assert len(teachers) > 0, "Teachers must be in %s" % " ".join(model_map.keys)

    # Initialize student model
    assert model_cfg["student"][0]['name'] in model_map, "Student must be in %s" % " ".join(model_map.keys)
    student = model_map[model_cfg["student"][0]['name']]() # eg: GaitSet()

    # Model setup
    device = torch.distributed.get_rank()
    torch.cuda.set_device(device)
    
    for i, teacher in enumerate(teachers):
        for p in teacher.parameters():
            p.requires_grad = False
        teacher = teacher.to(device=torch.device("cuda", device))
        teachers[i].__name__ = teacher.__name__

    # 加载预训练的teacher
    # Load parameters in teacher models
    for teacher in teachers:
        checkpoint = torch.load('./checkpoint/%s/ckpt.t7' % teacher.__name__)
        model_dict = teacher.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['net'].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        teacher.load_state_dict(model_dict)
        get_msg_mgr().log_info("teacher %s acc: ", (teacher.__name__, checkpoint['acc']))
    
    student = student.to(device=torch.device("cuda", device))
    if device == "cuda":
        out_dims = student.out_dims
        student = torch.nn.DataParallel(student)
        student.out_dims = out_dims

    if args.teacher_eval:
        for teacher in teachers:
            teacher.eval()

    return teachers, student
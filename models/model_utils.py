from models import *
import models.gaitgl as gaitgl
import models.gaitset as gaitset
import models.gaitpart as gaitpart
import models.baseline as baseline

def get_teachers_student(model_cfg):
    teachers = []

    model_map = {"GaitSet": gaitset.gaitSet,
                 "GaitPart": gaitpart.gaitPart,
                 "GaitGL": gaitgl.gaitGL,
                 'Baseline_ResNet9': baseline.baseline_ResNet9}

    # Add teachers models into teacher model list
    for t in model_cfg["teachers"]:
        if t in model_map:
            net = model_map[t]()  # eg: GaitSet()
            net.__name__ = t
            teachers.append(net)

    assert len(teachers) > 0, "Teachers must be in %s" % " ".join(model_map.keys)

    # Initialize student model

    assert model_cfg["student"] in model_map, "Student must be in %s" % " ".join(model_map.keys)
    student = model_map[model_cfg["student"]]() # eg: GaitSet()

    # Model setup

    if device == "cuda":
        cudnn.benchmark = True

    for i, teacher in enumerate(teachers):
        for p in teacher.parameters():
            p.requires_grad = False
        teacher = teacher.to(device)
        if device == "cuda":
            teachers[i] = torch.nn.DataParallel(teacher)
            teachers[i].__name__ = teacher.__name__

    # 加载预训练的teacher
    # Load parameters in teacher models
    for teacher in teachers:
        if teacher.__name__ != "shake_shake":
            checkpoint = torch.load('./checkpoint/%s/ckpt.t7' % teacher.__name__)
            model_dict = teacher.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['net'].items() if k in model_dict}
            model_dict.update(pretrained_dict)
            teacher.load_state_dict(model_dict)
            print("teacher %s acc: ", (teacher.__name__, checkpoint['acc']))

    student = student.to(device)
    if device == "cuda":
        out_dims = student.out_dims
        student = torch.nn.DataParallel(student)
        student.out_dims = out_dims

    if args.teacher_eval:
        for teacher in teachers:
            teacher.eval()

    return teachers, student
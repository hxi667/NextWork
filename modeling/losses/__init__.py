from inspect import isclass
from pkgutil import iter_modules
from pathlib import Path
from importlib import import_module
from .l1_soft import L1_soft
from .l2_soft import L2_soft
from .CE import CrossEntropy
from .KL import KL

import torch.nn.functional as F

'''
    Dynamically import all the modules in the current package,
    and add the classes in each module to the global variables of the current package.
'''

# Get the directory where the current script is located
package_dir = Path(__file__).resolve().parent
# Iterate all the modules in the current package
for (_, module_name, _) in iter_modules([str(package_dir)]):

    # Dynamic import of modules
    module = import_module(f"{__name__}.{module_name}")
    # Iterate the names of all attributes in the imported module
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)

        if isclass(attribute):
            # Add the class to the current package's global variables, keyed by the class name
            globals()[attribute_name] = attribute

loss_zoo = {"L1": F.l1_loss, # MAE
            "L2": F.mse_loss, # MSE
            "L1_soft": L1_soft,
            'L2_soft': L2_soft,
            'CE': CrossEntropy,
            'ED': F.pairwise_distance, # European distance
            'CS': F.cosine_similarity, # cosine_similarity
            'KL':KL, # KL dispersion
            } 


def get_loss(loss):
    if loss in loss_zoo:
        return loss_zoo[loss]
    else:
        raise NotImplementedError
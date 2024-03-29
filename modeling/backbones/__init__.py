from inspect import isclass
from pkgutil import iter_modules
from pathlib import Path
from importlib import import_module

'''
    动态地导入当前包中的所有modules，并将每个modules中的类添加到当前包的全局变量中
'''

# 获取当前脚本所在的目录
package_dir = Path(__file__).resolve().parent
# 遍历当前软件包中的所有modules
for (_, module_name, _) in iter_modules([str(package_dir)]):

    # 动态导入module
    module = import_module(f"{__name__}.{module_name}")
    # 遍历导入的module中的所有属性attributes的名称
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)

        if isclass(attribute):
            # 将该类添加到当前包的全局变量中，以类的名称为键
            globals()[attribute_name] = attribute
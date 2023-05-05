from Example.test.configs.l2_3 import *
from Example.test.configs.l4 import *
from Example.test.configs.l5 import *
from Example.test.configs.sens_l4 import *
from Example.test.configs.l2_3_l5 import *
from Example.test.configs.l4_l2_3 import *
from Example.test.configs.l5_l2_3 import *

if __name__ == '__main__':
    l2_3().save_as_yaml(file_name="config-snn", hard_refresh=True)
    l4().save_as_yaml(file_name="config-snn")
    l5().save_as_yaml(file_name="config-snn")
    sens_l4().save_as_yaml(file_name="config-snn")
    l2_3_l5().save_as_yaml(file_name="config-snn")
    l4_l2_3().save_as_yaml(file_name="config-snn")
    l5_l2_3().save_as_yaml(file_name="config-snn")

    l5_l2_3_instance = l5_l2_3()
    l5_l2_3_instance.load_as_yaml(file_name="config-snn")
    l5_l2_3_instance.exc_exc_structure_params['current_coef'] = 6
    l5_l2_3_instance.save_as_yaml(file_name="new-config")


from Example.test.configs.l2_3 import *
from Example.test.configs.l4 import *
from Example.test.configs.l5 import *
from Example.test.configs.sens_l4 import *
from Example.test.configs.l2_3_l5 import *
from Example.test.configs.l4_l2_3 import *
from Example.test.configs.l5_l2_3 import *

from conex.nn.config.base_config import BaseConfig

if __name__ == "__main__":
    config_type = "json"  # json
    l2_3().save(file_name=f"config-snn.{config_type}", hard_refresh=True)
    l4().save(file_name=f"config-snn.{config_type}")
    l5().save(file_name=f"config-snn.{config_type}")
    sens_l4().save(file_name=f"config-snn.{config_type}")
    l2_3_l5().save(file_name=f"config-snn.{config_type}")
    l4_l2_3().save(file_name=f"config-snn.{config_type}")
    l5_l2_3().save(file_name=f"config-snn.{config_type}")

    l5_l2_3_instance = l5_l2_3()
    l5_l2_3_instance.update_from_file(file_name=f"config-snn.{config_type}")
    l5_l2_3_instance.exc_exc_structure_params["current_coef"] = 6
    l5_l2_3_instance.save(file_name=f"new-config.{config_type}")

    loaded_instances = BaseConfig.load(file_name=f"config-snn.{config_type}")
    print(loaded_instances)
    print(loaded_instances["l2_3"].make())

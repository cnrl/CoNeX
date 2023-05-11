import importlib.util
import os
import collections.abc


from typing import Tuple, Union, Callable

from pymonntorch import *


class BaseConfig:
    def make(self):
        return {}

    @staticmethod
    def deep_update(main_dict, update_dict):
        for k, v in update_dict.items():
            if isinstance(v, collections.abc.Mapping):
                main_dict[k] = BaseConfig.deep_update(main_dict.get(k, {}), v)
            else:
                main_dict[k] = v
        return main_dict

    def update(self, update_dict):
        for key, value in update_dict.items():
            if isinstance(getattr(self, key), collections.abc.Mapping):
                setattr(self, key, BaseConfig.deep_update(getattr(self, key), value))
            else:
                setattr(self, key, value)

    def update_make(self, **kwargs):
        self.update(kwargs)
        return self.make()

    def __call__(self, *args, **kwargs):
        self.make(*args, **kwargs)

    def _get_members(self, sort_by):
        members = [
            attr
            for attr in dir(self)
            if attr
            not in [
                "update_from_yaml",
                "load_from_yaml",
                "save_as_yaml",
                "make",
                "update",
                "update_make",
                "deep_update",
            ]
            and not attr.startswith("_")
        ]
        members.sort(key=sort_by)
        return members

    @staticmethod
    def has_yaml_module():
        if importlib.util.find_spec('pyyaml') is None:
            raise ImportError("For using yaml file, you must have pyyaml=6.0 installed your environment!")

    def save_as_yaml(
        self,
        file_name,
        scope_key=None,
        configs_dir=".",
        sort_by=None,
        hard_refresh=False,
    ):
        """
        Args:
            file_name: file name where configs are going to be saved in.
            scope_key: scope where class config will be placed inside the yaml file, default is class name
            configs_dir: where configs are going to be saved
            sort_by: sorting algorithm used for ordering in the yaml configuration file
            hard_refresh: Pass True if you want to create new file under the same file_name

        Returns: None
        """
        self.has_yaml_module()
        import yaml

        members = self._get_members(sort_by)

        yaml_attributes_content = {attr: getattr(self, attr) for attr in members}

        scope_key = scope_key or self.__class__.__name__
        yaml_content = {
            scope_key: {
                "parameters": yaml_attributes_content,
                "class": {self.__class__.__name__: self.__class__},
            }
        }

        file_path = os.path.join(configs_dir, file_name)

        if os.path.isfile(file_path) and hard_refresh:
            os.remove(file_path)
            print(
                f"The file {file_name} has been deleted. A brand new config is going to be created!"
            )

        with open(file_path, "a") as yaml_file:
            yaml.dump(yaml_content, yaml_file, default_flow_style=False)

    def update_from_yaml(
        self, file_name, scope_key=None, configs_dir=".", force_update=False
    ):
        self.has_yaml_module()
        import yaml
        from yaml import Loader

        file_path = os.path.join(configs_dir, file_name)
        with open(file_path, "r") as yaml_file:
            yaml_content = yaml.load(yaml_file, Loader=Loader)

        scope_key = scope_key or self.__class__.__name__
        contents = yaml_content[scope_key]

        if not force_update:
            assert (
                self.__class__.__name__ in contents["class"]
            ), "YAML config should have been dumped from same class."

        self.update(contents["parameters"])

    @staticmethod
    def _make_config_instance(config):
        config_class = list(config["class"].values())[0]
        instance = config_class()
        instance.update(config["parameters"])
        return instance

    @staticmethod
    def load_from_yaml(file_name, configs_dir="."):
        BaseConfig.has_yaml_module()
        import yaml
        from yaml import Loader

        file_path = os.path.join(configs_dir, file_name)
        with open(file_path, "r") as yaml_file:
            yaml_content = yaml.load(yaml_file, Loader=Loader)

        configs = {
            scope: BaseConfig._make_config_instance(content)
            for scope, content in yaml_content.items()
        }

        return configs

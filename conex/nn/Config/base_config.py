import os
from typing import Tuple, Union, Callable

import yaml
import json
from pymonntorch import *

import collections.abc

from yaml import Loader


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
                "load_as_yaml",
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
        members = self._get_members(sort_by)

        yaml_attributes_content = {attr: getattr(self, attr) for attr in members}

        scope_key = scope_key or self.__class__.__name__
        yaml_content = {scope_key: yaml_attributes_content}

        if not file_name.endswith(".yml"):
            file_name += ".yml"

        file_path = os.path.join(configs_dir, file_name)

        if os.path.isfile(file_path) and hard_refresh:
            os.remove(file_path)
            print(
                f"The file {file_name} has been deleted. A brand new config is going to be created!"
            )

        with open(file_path, "a") as yaml_file:
            yaml.dump(yaml_content, yaml_file, default_flow_style=False)

    def load_as_yaml(self, file_name, scope_key=None, configs_dir="."):
        if not file_name.endswith(".yml"):
            file_name += ".yml"
        file_path = os.path.join(configs_dir, file_name)
        with open(file_path, "r") as yaml_file:
            yaml_content = yaml.load(yaml_file, Loader=Loader)

        scope_key = scope_key or self.__class__.__name__
        contents = yaml_content[scope_key]

        for attr, attr_value in contents.items():
            setattr(self, attr, attr_value)

import json
import os
import collections.abc
from typing import Callable

from pymonntorch import *

YML_EXT = ".yml"
JSON_EXT = ".json"


def object_hook(json_dict):
    def load_python_callables(raw_dict):
        for k, v in raw_dict.items():
            if isinstance(v, dict):
                load_python_callables(v)
            if isinstance(v, list) and len(v) > 0 and v[0] == "python_callable":
                import_string = f"import {v[1]}"
                exec(import_string)
                raw_dict[k] = eval(f"{v[1]}.{v[2]}")

    load_python_callables(json_dict)
    return json_dict


class ExtraCallableJSONEncoder(json.JSONEncoder):
    def default(self, object):
        if isinstance(object, Callable):
            return ("python_callable", object.__module__, object.__name__)
        super().default(object)


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
        public_members = [
            "save",
            "load",
            "update_from_file",
            "make",
            "update",
            "update_make",
            "deep_update",
        ]
        members = [
            attr
            for attr in dir(self)
            if (attr not in public_members and not attr.startswith("_"))
        ]
        members.sort(key=sort_by)
        return members

    def _store_content(
        self,
        file_name,
        scope_key=None,
        configs_dir=".",
        sort_by=None,
        hard_refresh=False,
        io_store_function=None,
        io_store_function_params=None,
    ):
        """
        Args:
            file_name: file name where configs are going to be saved in.
            scope_key: scope where class config will be placed inside the yaml file, default is class name
            configs_dir: where configs are going to be saved
            sort_by: sorting algorithm used for ordering in the yaml configuration file
            hard_refresh: Pass True if you want to create new file under the same file_name
            io_store_function: Function used to write content on io yaml.dump or json.dump
            io_store_function_params: Params that will be passed to the io_store_function

        Returns: None
        """
        members = self._get_members(sort_by)

        scope_key = scope_key or self.__class__.__name__
        data_content = {
            scope_key: {
                "parameters": {attr: getattr(self, attr) for attr in members},
                "class": {self.__class__.__name__: self.__class__},
            }
        }

        file_path = os.path.join(configs_dir, file_name)

        if os.path.isfile(file_path) and hard_refresh:
            os.remove(file_path)
            print(
                f"The file {file_name} has been deleted. A brand new config is going to be created!"
            )

        with open(file_path, "a+") as output_file:
            io_store_function_params = (
                {} if io_store_function_params is None else io_store_function_params
            )
            io_store_function(data_content, output_file, **io_store_function_params)

    def _update_content(
        self,
        file_name,
        scope_key=None,
        configs_dir=".",
        force_update=False,
        io_load_function=None,
        io_load_function_kwargs=None,
    ):
        file_path = os.path.join(configs_dir, file_name)
        with open(file_path, "r") as input_file:
            io_load_function_kwargs = (
                {} if io_load_function_kwargs is None else io_load_function_kwargs
            )
            data_content = io_load_function(input_file, **io_load_function_kwargs)

        scope_key = scope_key or self.__class__.__name__
        contents = data_content[scope_key]

        if not force_update:
            assert (
                self.__class__.__name__ in contents["class"]
            ), "Config should have been dumped from the same class."

        self.update(contents["parameters"])

    @staticmethod
    def _make_config_instance(config):
        config_class = list(config["class"].values())[0]
        instance = config_class()
        instance.update(config["parameters"])
        return instance

    @staticmethod
    def _load_content(
        file_name, configs_dir=".", io_load_function=None, io_load_function_kwargs=None
    ):
        file_path = os.path.join(configs_dir, file_name)
        with open(file_path, "r") as input_file:
            io_load_function_kwargs = (
                {} if io_load_function_kwargs is None else io_load_function_kwargs
            )
            data_content = io_load_function(input_file, **io_load_function_kwargs)

        configs = {
            scope: BaseConfig._make_config_instance(content)
            for scope, content in data_content.items()
        }

        return configs

    def _save_as_yaml(
        self,
        file_name,
        **store_content_kwargs,
    ):
        import yaml

        self._store_content(
            file_name,
            **store_content_kwargs,
            io_store_function=yaml.dump,
            io_store_function_params={"default_flow_style": False},
        )

    def _save_as_json(self, file_name, **store_content_kwargs):
        def multi_json_dump(data_content, output_file, **kwargs):
            output_file.seek(0)
            file_content = output_file.read()
            loaded_data = json.loads(file_content) if file_content else {}
            loaded_data = {**loaded_data, **data_content}
            output_file.truncate(0)
            json.dump(loaded_data, output_file, **kwargs)

        self._store_content(
            file_name,
            **store_content_kwargs,
            io_store_function=multi_json_dump,
            io_store_function_params={"cls": ExtraCallableJSONEncoder, "indent": 2},
        )

    def _update_from_json(self, file_name, **load_content_kwargs):
        self._update_content(
            file_name,
            **load_content_kwargs,
            io_load_function=json.load,
            io_load_function_kwargs={"object_hook": object_hook},
        )

    def _update_from_yaml(self, file_name, **load_content_kwargs):
        import yaml
        from yaml import Loader

        self._update_content(
            file_name,
            **load_content_kwargs,
            io_load_function=yaml.load,
            io_load_function_kwargs={"Loader": Loader},
        )

    @staticmethod
    def _load_from_yaml(file_name, **load_content_kwargs):
        import yaml
        from yaml import Loader

        return BaseConfig._load_content(
            file_name,
            **load_content_kwargs,
            io_load_function=yaml.load,
            io_load_function_kwargs={"Loader": Loader},
        )

    @staticmethod
    def _load_from_json(file_name, **load_content_kwargs):
        return BaseConfig._load_content(
            file_name,
            **load_content_kwargs,
            io_load_function=json.load,
            io_load_function_kwargs={"object_hook": object_hook},
        )

    def save(self, file_name, **store_content_kwargs):
        if file_name.endswith(YML_EXT):
            self._save_as_yaml(file_name, **store_content_kwargs)
        elif file_name.endswith(JSON_EXT):
            self._save_as_json(file_name, **store_content_kwargs)
        else:
            raise TypeError(f"{file_name} must end with .json or .yml")

    def update_from_file(self, file_name, **update_content_kwargs):
        if file_name.endswith(YML_EXT):
            self._update_from_yaml(file_name, **update_content_kwargs)
        elif file_name.endswith(JSON_EXT):
            self._update_from_json(file_name, **update_content_kwargs)
        else:
            raise TypeError(f"{file_name} must end with .json or .yaml")

    @staticmethod
    def load(file_name, **load_content_kwargs):
        if file_name.endswith(YML_EXT):
            return BaseConfig._load_from_yaml(file_name, **load_content_kwargs)
        elif file_name.endswith(JSON_EXT):
            return BaseConfig._load_from_json(file_name, **load_content_kwargs)
        else:
            raise TypeError(f"{file_name} must end with .json or .yaml")

from typing import Tuple, Union
from collections.abc import Mapping
from .base_config import *

from pymonntorch import *

# TODO: check default valus


class Layer2LayerConnectionConfig(BaseConfig):
    exc_exc_weight_init_params = {}
    exc_exc_structure = "Simple"
    exc_exc_structure_params = {}
    exc_exc_learning_rule = None
    exc_exc_learning_params = None
    exc_exc_src_delay_init_mode = None
    exc_exc_src_delay_init_params = None
    exc_exc_dst_delay_init_mode = None
    exc_exc_dst_delay_init_params = None
    exc_exc_w_interval = None
    exc_exc_weight_norm = None
    exc_exc_tag = None
    exc_exc_user_defined_behaviors_class = None
    exc_exc_user_defined_behaviors_params = None
    exc_exc_src_pop = "exc_pop"
    exc_exc_dst_pop = "exc_pop"

    exc_inh_weight_init_params = {}
    exc_inh_structure = "Simple"
    exc_inh_structure_params = {}
    exc_inh_learning_rule = None
    exc_inh_learning_params = None
    exc_inh_src_delay_init_mode = None
    exc_inh_src_delay_init_params = None
    exc_inh_dst_delay_init_mode = None
    exc_inh_dst_delay_init_params = None
    exc_inh_w_interval = None
    exc_inh_weight_norm = None
    exc_inh_tag = None
    exc_inh_user_defined_behaviors_class = None
    exc_inh_user_defined_behaviors_params = None
    exc_inh_src_pop = "exc_pop"
    exc_inh_dst_pop = "inh_pop"

    inh_exc_weight_init_params = {}
    inh_exc_structure = "Simple"
    inh_exc_structure_params = {}
    inh_exc_learning_rule = None
    inh_exc_learning_params = None
    inh_exc_src_delay_init_mode = None
    inh_exc_src_delay_init_params = None
    inh_exc_dst_delay_init_mode = None
    inh_exc_dst_delay_init_params = None
    inh_exc_w_interval = None
    inh_exc_weight_norm = None
    inh_exc_tag = None
    inh_exc_user_defined_behaviors_class = None
    inh_exc_user_defined_behaviors_params = None
    inh_exc_src_pop = "inh_pop"
    inh_exc_dst_pop = "exc_pop"

    inh_inh_weight_init_params = {}
    inh_inh_structure = "Simple"
    inh_inh_structure_params = {}
    inh_inh_learning_rule = None
    inh_inh_learning_params = None
    inh_inh_src_delay_init_mode = None
    inh_inh_src_delay_init_params = None
    inh_inh_dst_delay_init_mode = None
    inh_inh_dst_delay_init_params = None
    inh_inh_w_interval = None
    inh_inh_weight_norm = None
    inh_inh_tag = None
    inh_inh_user_defined_behaviors_class = None
    inh_inh_user_defined_behaviors_params = None
    inh_inh_src_pop = "inh_pop"
    inh_inh_dst_pop = "inh_pop"

    @classmethod
    def _syn_config(
        cls,
        weight_init_params: Mapping,
        structure: Union[str, type[Behavior]],
        structure_params: Mapping,
        learning_rule: Union[None, str, type[Behavior]] = None,
        learning_params: Union[None, Mapping] = None,
        src_delay_init_mode: Union[None, float, str] = None,
        src_delay_init_params: Union[None, Mapping] = None,
        dst_delay_init_mode: Union[None, float, str] = None,
        dst_delay_init_params: Union[None, Mapping] = None,
        w_interval: Union[None, Tuple[float, float]] = None,
        weight_norm: Union[None, float] = None,
        tag: Union[None, str] = None,
        user_defined_behaviors_class: Union[None, Mapping[int, type[Behavior]]] = None,
        user_defined_behaviors_params: Union[None, Mapping[int, Mapping]] = None,
        src_pop: Union[None, str] = None,
        dst_pop: Union[None, str] = None,
    ) -> Mapping:
        config = {
            "weight_init_params": weight_init_params,
            "structure": structure,
            "structure_params": structure_params,
        }

        if learning_rule is not None:
            config["learning_rule"] = learning_rule
            if learning_params is None:
                learning_params = {}
            config["learning_params"] = learning_params

        if src_delay_init_mode is not None:
            config["src_delay_init_mode"] = src_delay_init_mode
            if src_delay_init_params is None:
                src_delay_init_params = {}
            config["src_delay_init_params"] = src_delay_init_params

        if dst_delay_init_mode is not None:
            config["dst_delay_init_mode"] = dst_delay_init_mode
            if dst_delay_init_params is None:
                dst_delay_init_params = {}
            config["dst_delay_init_params"] = dst_delay_init_params

        if w_interval is not None:
            config["w_interval"] = w_interval

        if weight_norm is not None:
            config["weight_norm"] = weight_norm

        if tag is not None:
            config["tag"] = tag

        if user_defined_behaviors_class is not None:
            config["user_defined"] = {}
            for k, v in user_defined_behaviors_class.items():
                params = (
                    user_defined_behaviors_params.get(k, {})
                    if user_defined_behaviors_params is not None
                    else {}
                )
                config["user_defined"][k] = v(**params)

        if src_pop is not None:
            config["src_pop"] = src_pop

        if dst_pop is not None:
            config["dst_pop"] = dst_pop

        return config

    def make(self):
        config = {}

        if self.exc_exc_weight_init_params:
            config["exc_exc"] = self._syn_config(
                self.exc_exc_weight_init_params,
                self.exc_exc_structure,
                self.exc_exc_structure_params,
                self.exc_exc_learning_rule,
                self.exc_exc_learning_params,
                self.exc_exc_src_delay_init_mode,
                self.exc_exc_src_delay_init_params,
                self.exc_exc_dst_delay_init_mode,
                self.exc_exc_dst_delay_init_params,
                self.exc_exc_w_interval,
                self.exc_exc_weight_norm,
                self.exc_exc_tag,
                self.exc_exc_user_defined_behaviors_class,
                self.exc_exc_user_defined_behaviors_params,
                self.exc_exc_src_pop,
                self.exc_exc_dst_pop,
            )

        if self.exc_inh_weight_init_params:
            config["exc_inh"] = self._syn_config(
                self.exc_inh_weight_init_params,
                self.exc_inh_structure,
                self.exc_inh_structure_params,
                self.exc_inh_learning_rule,
                self.exc_inh_learning_params,
                self.exc_inh_src_delay_init_mode,
                self.exc_inh_src_delay_init_params,
                self.exc_inh_dst_delay_init_mode,
                self.exc_inh_dst_delay_init_params,
                self.exc_inh_w_interval,
                self.exc_inh_weight_norm,
                self.exc_inh_tag,
                self.exc_inh_user_defined_behaviors_class,
                self.exc_inh_user_defined_behaviors_params,
                self.exc_inh_src_pop,
                self.exc_inh_dst_pop,
            )

        if self.inh_exc_weight_init_params:
            config["inh_exc"] = self._syn_config(
                self.inh_exc_weight_init_params,
                self.inh_exc_structure,
                self.inh_exc_structure_params,
                self.inh_exc_learning_rule,
                self.inh_exc_learning_params,
                self.inh_exc_src_delay_init_mode,
                self.inh_exc_src_delay_init_params,
                self.inh_exc_dst_delay_init_mode,
                self.inh_exc_dst_delay_init_params,
                self.inh_exc_w_interval,
                self.inh_exc_weight_norm,
                self.inh_exc_tag,
                self.inh_exc_user_defined_behaviors_class,
                self.inh_exc_user_defined_behaviors_params,
                self.inh_exc_src_pop,
                self.inh_exc_dst_pop,
            )

        if self.inh_inh_weight_init_params:
            config["inh_inh"] = self._syn_config(
                self.inh_inh_weight_init_params,
                self.inh_inh_structure,
                self.inh_inh_structure_params,
                self.inh_inh_learning_rule,
                self.inh_inh_learning_params,
                self.inh_inh_src_delay_init_mode,
                self.inh_inh_src_delay_init_params,
                self.inh_inh_dst_delay_init_mode,
                self.inh_inh_dst_delay_init_params,
                self.inh_inh_w_interval,
                self.inh_inh_weight_norm,
                self.inh_inh_tag,
                self.inh_inh_user_defined_behaviors_class,
                self.inh_inh_user_defined_behaviors_params,
                self.inh_inh_src_pop,
                self.inh_inh_dst_pop,
            )

        return config


class Pop2LayerConnectionConfig(BaseConfig):
    pop_2_exc_weight_init_params = {}
    pop_2_exc_structure = "Simple"
    pop_2_exc_structure_params = {}
    pop_2_exc_learning_rule = None
    pop_2_exc_learning_params = None
    pop_2_exc_src_delay_init_mode = None
    pop_2_exc_src_delay_init_params = None
    pop_2_exc_dst_delay_init_mode = None
    pop_2_exc_dst_delay_init_params = None
    pop_2_exc_w_interval = None
    pop_2_exc_weight_norm = None
    pop_2_exc_tag = None
    pop_2_exc_user_defined_behaviors_class = None
    pop_2_exc_user_defined_behaviors_params = None
    pop_2_exc_dst_pop = "exc_pop"

    pop_2_inh_weight_init_params = {}
    pop_2_inh_structure = "Simple"
    pop_2_inh_structure_params = {}
    pop_2_inh_learning_rule = None
    pop_2_inh_learning_params = None
    pop_2_inh_src_delay_init_mode = None
    pop_2_inh_src_delay_init_params = None
    pop_2_inh_dst_delay_init_mode = None
    pop_2_inh_dst_delay_init_params = None
    pop_2_inh_w_interval = None
    pop_2_inh_weight_norm = None
    pop_2_inh_tag = None
    pop_2_inh_user_defined_behaviors_class = None
    pop_2_inh_user_defined_behaviors_params = None
    pop_2_inh_dst_pop = "inh_pop"

    @classmethod
    def _syn_config(
        cls,
        weight_init_params: Mapping,
        structure: Union[str, type[Behavior]],
        structure_params: Mapping,
        learning_rule: Union[None, str, type[Behavior]] = None,
        learning_params: Union[None, Mapping] = None,
        src_delay_init_mode: Union[None, float, str] = None,
        src_delay_init_params: Union[None, Mapping] = None,
        dst_delay_init_mode: Union[None, float, str] = None,
        dst_delay_init_params: Union[None, Mapping] = None,
        w_interval: Union[None, Tuple[float, float]] = None,
        weight_norm: Union[None, float] = None,
        tag: Union[None, str] = None,
        user_defined_behaviors_class: Union[None, Mapping[int, type[Behavior]]] = None,
        user_defined_behaviors_params: Union[None, Mapping[int, Mapping]] = None,
        dst_pop: Union[None, str] = None,
    ) -> Mapping:
        config = {
            "weight_init_params": weight_init_params,
            "structure": structure,
            "structure_params": structure_params,
        }

        if learning_rule is not None:
            config["learning_rule"] = learning_rule
            if learning_params is None:
                learning_params = {}
            config["learning_params"] = learning_params

        if src_delay_init_mode is not None:
            config["src_delay_init_mode"] = src_delay_init_mode
            if src_delay_init_params is None:
                src_delay_init_params = {}
            config["src_delay_init_params"] = src_delay_init_params

        if dst_delay_init_mode is not None:
            config["dst_delay_init_mode"] = dst_delay_init_mode
            if dst_delay_init_params is None:
                dst_delay_init_params = {}
            config["dst_delay_init_params"] = dst_delay_init_params

        if w_interval is not None:
            config["w_interval"] = w_interval

        if weight_norm is not None:
            config["weight_norm"] = weight_norm

        if tag is not None:
            config["tag"] = tag

        if user_defined_behaviors_class is not None:
            config["user_defined"] = {}
            for k, v in user_defined_behaviors_class.items():
                params = (
                    user_defined_behaviors_params.get(k, {})
                    if user_defined_behaviors_params is not None
                    else {}
                )
                config["user_defined"][k] = v(**params)

        if dst_pop is not None:
            config["dst_pop"] = dst_pop

        return config

    def make(self):
        config = {}

        if self.pop_2_exc_weight_init_params:
            config["pop_2_exc"] = self._syn_config(
                self.pop_2_exc_weight_init_params,
                self.pop_2_exc_structure,
                self.pop_2_exc_structure_params,
                self.pop_2_exc_learning_rule,
                self.pop_2_exc_learning_params,
                self.pop_2_exc_src_delay_init_mode,
                self.pop_2_exc_src_delay_init_params,
                self.pop_2_exc_dst_delay_init_mode,
                self.pop_2_exc_dst_delay_init_params,
                self.pop_2_exc_w_interval,
                self.pop_2_exc_weight_norm,
                self.pop_2_exc_tag,
                self.pop_2_exc_user_defined_behaviors_class,
                self.pop_2_exc_user_defined_behaviors_params,
                self.pop_2_exc_dst_pop,
            )

        if self.pop_2_inh_weight_init_params:
            config["pop_2_inh"] = self._syn_config(
                self.pop_2_inh_weight_init_params,
                self.pop_2_inh_structure,
                self.pop_2_inh_structure_params,
                self.pop_2_inh_learning_rule,
                self.pop_2_inh_learning_params,
                self.pop_2_inh_src_delay_init_mode,
                self.pop_2_inh_src_delay_init_params,
                self.pop_2_inh_dst_delay_init_mode,
                self.pop_2_inh_dst_delay_init_params,
                self.pop_2_inh_w_interval,
                self.pop_2_inh_weight_norm,
                self.pop_2_inh_tag,
                self.pop_2_inh_user_defined_behaviors_class,
                self.pop_2_inh_user_defined_behaviors_params,
                self.pop_2_inh_dst_pop,
            )

        return config


class Layer2PopConnectionConfig(BaseConfig):
    exc_2_pop_weight_init_params = {}
    exc_2_pop_structure = "Simple"
    exc_2_pop_structure_params = {}
    exc_2_pop_learning_rule = None
    exc_2_pop_learning_params = None
    exc_2_pop_src_delay_init_mode = None
    exc_2_pop_src_delay_init_params = None
    exc_2_pop_dst_delay_init_mode = None
    exc_2_pop_dst_delay_init_params = None
    exc_2_pop_w_interval = None
    exc_2_pop_weight_norm = None
    exc_2_pop_tag = None
    exc_2_pop_user_defined_behaviors_class = None
    exc_2_pop_user_defined_behaviors_params = None
    exc_2_pop_src_pop = "exc_pop"

    inh_2_pop_weight_init_params = {}
    inh_2_pop_structure = "Simple"
    inh_2_pop_structure_params = {}
    inh_2_pop_learning_rule = None
    inh_2_pop_learning_params = None
    inh_2_pop_src_delay_init_mode = None
    inh_2_pop_src_delay_init_params = None
    inh_2_pop_dst_delay_init_mode = None
    inh_2_pop_dst_delay_init_params = None
    inh_2_pop_w_interval = None
    inh_2_pop_weight_norm = None
    inh_2_pop_tag = None
    inh_2_pop_user_defined_behaviors_class = None
    inh_2_pop_user_defined_behaviors_params = None
    inh_2_pop_src_pop = "inh_pop"

    @classmethod
    def _syn_config(
        cls,
        weight_init_params: Mapping,
        structure: Union[str, type[Behavior]],
        structure_params: Mapping,
        learning_rule: Union[None, str, type[Behavior]] = None,
        learning_params: Union[None, Mapping] = None,
        src_delay_init_mode: Union[None, float, str] = None,
        src_delay_init_params: Union[None, Mapping] = None,
        dst_delay_init_mode: Union[None, float, str] = None,
        dst_delay_init_params: Union[None, Mapping] = None,
        w_interval: Union[None, Tuple[float, float]] = None,
        weight_norm: Union[None, float] = None,
        tag: Union[None, str] = None,
        user_defined_behaviors_class: Union[None, Mapping[int, type[Behavior]]] = None,
        user_defined_behaviors_params: Union[None, Mapping[int, Mapping]] = None,
        src_pop: Union[None, str] = None,
    ) -> Mapping:
        config = {
            "weight_init_params": weight_init_params,
            "structure": structure,
            "structure_params": structure_params,
        }

        if learning_rule is not None:
            config["learning_rule"] = learning_rule
            if learning_params is None:
                learning_params = {}
            config["learning_params"] = learning_params

        if src_delay_init_mode is not None:
            config["src_delay_init_mode"] = src_delay_init_mode
            if src_delay_init_params is None:
                src_delay_init_params = {}
            config["src_delay_init_params"] = src_delay_init_params

        if dst_delay_init_mode is not None:
            config["dst_delay_init_mode"] = dst_delay_init_mode
            if dst_delay_init_params is None:
                dst_delay_init_params = {}
            config["dst_delay_init_params"] = dst_delay_init_params

        if w_interval is not None:
            config["w_interval"] = w_interval

        if weight_norm is not None:
            config["weight_norm"] = weight_norm

        if tag is not None:
            config["tag"] = tag

        if user_defined_behaviors_class is not None:
            config["user_defined"] = {}
            for k, v in user_defined_behaviors_class.items():
                params = (
                    user_defined_behaviors_params.get(k, {})
                    if user_defined_behaviors_params is not None
                    else {}
                )
                config["user_defined"][k] = v(**params)

        if src_pop is not None:
            config["src_pop"] = src_pop

        return config

    def make(self):
        config = {}

        if self.exc_2_pop_weight_init_params:
            config["exc_2_pop"] = self._syn_config(
                self.exc_2_pop_weight_init_params,
                self.exc_2_pop_structure,
                self.exc_2_pop_structure_params,
                self.exc_2_pop_learning_rule,
                self.exc_2_pop_learning_params,
                self.exc_2_pop_src_delay_init_mode,
                self.exc_2_pop_src_delay_init_params,
                self.exc_2_pop_dst_delay_init_mode,
                self.exc_2_pop_dst_delay_init_params,
                self.exc_2_pop_w_interval,
                self.exc_2_pop_weight_norm,
                self.exc_2_pop_tag,
                self.exc_2_pop_user_defined_behaviors_class,
                self.exc_2_pop_user_defined_behaviors_params,
                self.exc_2_pop_src_pop,
            )

        if self.inh_2_pop_weight_init_params:
            config["inh_2_pop"] = self._syn_config(
                self.inh_2_pop_weight_init_params,
                self.inh_2_pop_structure,
                self.inh_2_pop_structure_params,
                self.inh_2_pop_learning_rule,
                self.inh_2_pop_learning_params,
                self.inh_2_pop_src_delay_init_mode,
                self.inh_2_pop_src_delay_init_params,
                self.inh_2_pop_dst_delay_init_mode,
                self.inh_2_pop_dst_delay_init_params,
                self.inh_2_pop_w_interval,
                self.inh_2_pop_weight_norm,
                self.inh_2_pop_tag,
                self.inh_2_pop_user_defined_behaviors_class,
                self.inh_2_pop_user_defined_behaviors_params,
                self.inh_2_pop_src_pop,
            )

        return config
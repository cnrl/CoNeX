from typing import Tuple, Union, Callable
from .base_config import *
from pymonntorch import *


class ConnectionConfig(BaseConfig):
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
    exc_exc_src_pop = None
    exc_exc_dst_pop = None

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
    exc_inh_src_pop = None
    exc_inh_dst_pop = None

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
    inh_exc_src_pop = None
    inh_exc_dst_pop = None

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
    inh_inh_src_pop = None
    inh_inh_dst_pop = None

    @classmethod
    def _syn_config(
        cls,
        weight_init_params: dict,
        structure: Union[str, type[Behavior]],
        structure_params: dict,
        learning_rule: Union[None, str, type[Behavior]] = None,
        learning_params: Union[None, dict] = None,
        src_delay_init_mode: Union[None, float, str] = None,
        src_delay_init_params: Union[None, dict] = None,
        dst_delay_init_mode: Union[None, float, str] = None,
        dst_delay_init_params: Union[None, dict] = None,
        w_interval: Union[None, Tuple[float, float]] = None,
        weight_norm: Union[None, float] = None,
        tag: Union[None, str] = None,
        user_defined_behaviors_class: Union[None, dict[int, type[Behavior]]] = None,
        user_defined_behaviors_params: Union[None, dict[int, dict]] = None,
        src_pop: Union[None, str] = None,
        dst_pop: Union[None, str] = None,
    ) -> dict:
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

import copy
from pymonntorch import NeuronGroup, SynapseGroup, NetworkObject, Behavior, Network
from typing import Dict, List
from conex.nn.structure.port import Port
from conex.nn.structure.io_layer import InputLayer, OutputLayer

ELEMENTAL_STRUCTURE = [NeuronGroup, SynapseGroup, InputLayer, OutputLayer]


def get_all_required_structures(struc: NetworkObject) -> List[NetworkObject]:
    net = struc.network
    all_required_sub_structures = [struc.required_helper()]
    while all_required_sub_structures[-1]:
        current_struc = []
        for x in all_required_sub_structures[-1]:
            if x.__class__ not in ELEMENTAL_STRUCTURE:
                current_struc.extend(current_struc.required_helper())
        all_required_sub_structures.append(current_struc)

    flatten = [x for xs in all_required_sub_structures for x in xs]
    if len(flatten) != len(set(flatten)):
        print("ERROR: duplicate elements returned by required_helpers")

    return sorted(flatten, key=lambda x: net.Structures.index(x))


def save_ports(ports, save_behavior_tag: bool, save_behavior_priority: bool) -> dict:
    result = {}
    for key, value in ports.items():
        result[key] = [
            (
                x[0],
                x[1],
                behaviors_to_list(
                    x[2],
                    save_behavior_priority=save_behavior_priority,
                    save_behavior_tag=save_behavior_tag,
                ),
            )
            for x in value
        ]
    return result


def build_ports(ports: Dict[str, List[Port]]) -> Dict[str, tuple]:
    result = {}
    for key, value in ports.items():
        result[key] = [(x[0], x[1], build_behavior_dict(x[2])) for x in value]
    return result


def save_structure(
    struc: NetworkObject,
    save_device: bool = False,
    built_structures: dict = None,
    save_structure_tag: bool = False,
    save_behavior_tag: bool = True,
    save_behavior_priority: bool = True,
    all_structures_required: list = None,
) -> dict:
    result = {"tag": None}
    result["class"] = struc.__class__
    result["behavior"] = behaviors_to_list(
        struc.behavior,
        save_behavior_tag=save_behavior_tag,
        save_behavior_priority=save_behavior_priority,
    )

    if save_device:
        result["device"] = struc.device

    if save_structure_tag:
        result["tag"] = ",".join(struc.tags)

    if all_structures_required is None:
        result["built_structures"] = {}
        built_structures = result["built_structures"]
        all_structures_required = get_all_required_structures(struc)

    if isinstance(struc, NeuronGroup):
        result["size"] = struc.size
    elif isinstance(struc, SynapseGroup):
        result["src"] = (
            all_structures_required.index(struc.src) if struc.src is not None else None
        )
        result["dst"] = (
            all_structures_required.index(struc.dst) if struc.src is not None else None
        )
    elif isinstance(struc, InputLayer) or isinstance(struc, OutputLayer):
        print("ERROR: InputLayer and OutputLayer can't be saved.")
    else:
        for sub_struc in all_structures_required:
            if all_structures_required.index(sub_struc) not in built_structures:
                built_sub_struc = save_structure(
                    sub_struc,
                    all_structures_required=all_structures_required,
                    built_structures=built_structures,
                    save_structure_tag=save_structure_tag,
                    save_behavior_tag=save_behavior_tag,
                    save_behavior_priority=save_behavior_priority,
                    save_device=save_device,
                )
                built_structures[
                    all_structures_required.index(sub_struc)
                ] = built_sub_struc

        helper_dictionary = struc.save_helper(all_structures_required)
        result.update(helper_dictionary)

    return result


def behaviors_to_list(
    behavior: Dict[int, Behavior],
    save_behavior_tag: bool = True,
    save_behavior_priority: bool = True,
) -> List[dict]:
    result = []
    for key, value in behavior.items():
        behave_dict = {}
        if save_behavior_priority:
            behave_dict["key"] = key
        behave_dict["class"] = value.__class__
        args = {
            int(k[4:]): val
            for k, val in value.init_kwargs.items()
            if k.startswith("arg_")
        }
        behave_dict["parameters_args"] = [args[i] for i in sorted(list(args.keys()))]
        behave_dict["parameters_kwargs"] = {
            k: val for k, val in value.init_kwargs.items() if not k.startswith("arg_")
        }
        if not save_behavior_tag:
            _ = behave_dict["parameters_kwargs"].pop("tag", "")
        result.append(behave_dict)

    return result


def build_behavior_dict(behavior_list: List[dict]) -> Dict[int, Behavior]:
    result = {}
    for i, beh_dict in enumerate(behavior_list):
        key = i if "key" not in beh_dict else beh_dict["key"]
        value = beh_dict["class"](
            *beh_dict["parameters_args"], **beh_dict["parameters_kwargs"]
        )
        result[key] = value
    return result


def create_structure_from_dict(
    net: Network, structure_dict: dict, built_structures: dict = None
) -> NetworkObject:
    struc_dict = copy.deepcopy(structure_dict)
    if struc_dict["class"] == NeuronGroup:
        return NeuronGroup(
            struc_dict["size"],
            net=net,
            behavior=build_behavior_dict(struc_dict["behavior"]),
            tag=struc_dict["tag"],
        )

    if struc_dict["class"] == SynapseGroup:
        print(built_structures)
        return SynapseGroup(
            net=net,
            src=struc_dict["src"]
            if struc_dict["src"] is None
            else built_structures[struc_dict["src"]],
            dst=struc_dict["dst"]
            if struc_dict["dst"] is None
            else built_structures[struc_dict["dst"]],
            behavior=build_behavior_dict(struc_dict["behavior"]),
            tag=struc_dict["tag"],
        )

    if struc_dict["class"] == InputLayer:
        print("ERROR: recived InputLayer, Which cannot be saved.")
        return None

    if struc_dict["class"] == OutputLayer:
        print("ERROR: recived OutputLayer, Which cannot be saved.")
        return None

    if built_structures is None:
        built_structures = {}

    required_structures = struc_dict.pop("built_structures", {})
    for k, v in required_structures.items():
        if k not in built_structures:
            built_structures[k] = create_structure_from_dict(net, v, built_structures)
        else:
            print("ERROR: found duplicate in required structure.")

    class_obj = struc_dict.pop("class")
    _ = class_obj.build_helper(struc_dict, built_structures)

    # for extra in []:
    #     helper_dict.pop(extra)

    # build_helper might intentionally change the behavior.
    struc_dict["behavior"] = build_behavior_dict(struc_dict["behavior"])

    return class_obj(net=net, **struc_dict)


def replicate(obj: NetworkObject, net: Network) -> NetworkObject:
    save_dict = save_structure(
        obj,
        save_device=True,
        built_structures=None,
        save_structure_tag=True,
        save_behavior_tag=True,
        save_behavior_priority=True,
        all_structures_required=None,
    )
    new_dict = copy.deepcopy(save_dict)
    struc = create_structure_from_dict(
        net=net, structure_dict=new_dict, built_structures=None
    )
    return struc, save_dict

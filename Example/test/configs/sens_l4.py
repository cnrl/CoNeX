from conex.nn.Config.connection_config import Input2LayerConnectionConfig


class sens_l4(Input2LayerConnectionConfig):
    exc_weight_init_params = {"mode": "uniform", "weight_shape": [4, 1, 4, 4]}
    exc_structure = "Conv2d"
    exc_structure_params = {"current_coef": 3}
    exc_dst_pop = "exc_pop"
    exc_tag = "Proximal"

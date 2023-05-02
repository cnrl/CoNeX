from conex.nn.Config.connection_config import Pop2LayerConnectionConfig


class sens_l4(Pop2LayerConnectionConfig):
    pop_2_exc_weight_init_params = {"mode": "uniform", "weight_shape": [4, 1, 4, 4]}
    pop_2_exc_structure = "Conv2d"
    pop_2_exc_structure_params = {"current_coef": 3}
    pop_2_exc_dst_pop = "exc_pop"
    pop_2_exc_tag = "Proximal"

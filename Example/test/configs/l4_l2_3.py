from conex.nn.Config.connection_config import Layer2LayerConnectionConfig


class l4_l2_3(Layer2LayerConnectionConfig):
    exc_exc_weight_init_params = {"mode": "uniform"}
    exc_exc_structure = "Simple"
    exc_exc_structure_params = {"current_coef": 3}
    exc_exc_src_pop = "exc_pop"
    exc_exc_dst_pop = "exc_pop"

from conex.nn.Config.connection_config import Pop2LayerConnectionConfig


class l2_3_l5(Pop2LayerConnectionConfig):
    exc_exc_weight_init_params = {"mode": "uniform"}
    exc_exc_structure = "Simple"
    exc_exc_structure_params = {"current_coef": 3}
    exc_exc_src_pop = "exc_pop"
    exc_exc_dst_pop = "exc_pop"

from conex.nn.Config.connection_config import Layer2PopConnectionConfig


class l2_3_rep(Layer2PopConnectionConfig):
    exc_2_pop_weight_init_params = {"mode": "uniform"}
    exc_2_pop_structure = "Simple"
    exc_2_pop_structure_params = {"current_coef": 3}
    exc_2_pop_src_pop = "exc_pop"
    exc_2_pop_tag = "Proximal"

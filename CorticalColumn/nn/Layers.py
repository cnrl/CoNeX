"""
Implementation of Cortical layers.
"""
from CorticalColumn.behaviours.neurons.lif_neurons import AELIF, LIF
from CorticalColumn.behaviours.neurons.specs import KWTA, SpikeHistory, SpikeTrace
from CorticalColumn.behaviours.synapses.dendrites import (
    ApicalInput,
    BasalInput,
    ProximalInput,
)
from CorticalColumn.behaviours.synapses.learning import STDP
from CorticalColumn.behaviours.synapses.specs import WeightClip
from CorticalColumn.nn.Modules.spiking_neurons import SpikingNeuronGroup
from CorticalColumn.nn.Modules.topological_connections import SparseSynapseGroup

# TODO: Complete implementation of each layer, including the undefined methods and attributes.


class L2_3:
    def __init__(
        self,
        exc_shape,
        inh_shape,
        net,
        exc_exc_rate=0.2,
        exc_inh_rate=0.4,
        inh_exc_rate=0.4,
        inh_inh_rate=0.1,
    ):
        self.exc_pop = SpikingNeuronGroup(
            exc_shape,
            behaviour={
                1: KWTA(k=10),
                3: AELIF(tau=20, r=1, delta=2, theta_rh=15, alpha=1, beta=1.5),
                4: SpikeTrace(),
                5: SpikeHistory(),
            },
            net=net,
            v_rest=0.0,
            v_reset=-2.0,
            threshold=20,
            tau_s=40.0,
        )
        self.inh_pop = SpikingNeuronGroup(
            inh_shape,
            behaviour={
                3: LIF(tau=20, r=1, delta=2, theta_rh=6, alpha=1, beta=1.5),
                4: SpikeTrace(),
                5: SpikeHistory(),
            },
            net=net,
            v_rest=0.0,
            v_reset=-2.0,
            threshold=10,
            tau_s=30.0,
        )

        self.exc_exc_conn = SparseSynapseGroup(
            self.exc_pop,
            self.exc_pop,
            net,
            exc_exc_rate,
            weight_init_mode="normal(0.25, 0.25)",
            delay_init_mode="normal(0.5, 0.5)",
            behaviour={
                3: ApicalInput(),
                4: BasalInput(),
                5: ProximalInput(),
                6: STDP(a_plus=0.005, a_minus=0.002),
                7: WeightClip(w_min=0.0, w_max=0.5),
            },
        )
        self.exc_inh_conn = SparseSynapseGroup(
            self.exc_pop,
            self.inh_pop,
            net,
            exc_inh_rate,
            weight_init_mode="normal(0.5, 0.5)",
            delay_init_mode="normal(0.5, 0.5)",
            behaviour={
                3: ApicalInput(),
                4: BasalInput(),
                5: ProximalInput(),
                6: STDP(a_plus=0.05, a_minus=0.02),
                7: WeightClip(w_min=0.0, w_max=1.0),
            },
        )
        self.inh_exc_conn = SparseSynapseGroup(
            self.inh_pop,
            self.exc_pop,
            net,
            inh_exc_rate,
            weight_init_mode="normal(0.5, 0.5)",
            delay_init_mode="normal(0.5, 0.5)",
            behaviour={
                3: ApicalInput(),
                4: BasalInput(),
                5: ProximalInput(),
                6: STDP(a_plus=0.02, a_minus=0.01),
                7: WeightClip(w_min=0.0, w_max=0.5),
            },
        )
        self.inh_inh_conn = SparseSynapseGroup(
            self.inh_pop,
            self.inh_pop,
            net,
            inh_inh_rate,
            weight_init_mode="normal(0.25, 0.25)",
            delay_init_mode="normal(0.5, 0.5)",
            behaviour={
                3: ApicalInput(),
                4: BasalInput(),
                5: ProximalInput(),
                6: STDP(a_plus=0.002, a_minus=0.001),
                7: WeightClip(w_min=0.0, w_max=0.5),
            },
        )

    def forward(self, input):
        pass

    def backward(self, delta):
        pass

    def update(self, learning_rate):
        pass


class L4:
    def __init__(
        self,
        exc_shape,
        inh_shape,
        net,
        exc_exc_rate=0.2,
        exc_inh_rate=0.4,
        inh_exc_rate=0.4,
        inh_inh_rate=0.1,
    ):
        self.exc_pop = SpikingNeuronGroup(
            exc_shape,
            behaviour={
                3: AELIF(tau=20, r=1, delta=2, theta_rh=15, alpha=1, beta=1.5),
                4: SpikeTrace(),
                5: SpikeHistory(),
            },
            net=net,
            v_rest=0.0,
            v_reset=-2.0,
            threshold=20,
            tau_s=40.0,
        )
        self.inh_pop = SpikingNeuronGroup(
            inh_shape,
            behaviour={
                3: LIF(tau=20, r=1, delta=2, theta_rh=6, alpha=1, beta=1.5),
                4: SpikeTrace(),
                5: SpikeHistory(),
            },
            net=net,
            v_rest=0.0,
            v_reset=-2.0,
            threshold=10,
            tau_s=30.0,
        )

        self.exc_exc_conn = SparseSynapseGroup(
            self.exc_pop,
            self.exc_pop,
            net,
            exc_exc_rate,
            weight_init_mode="normal(0.25, 0.25)",
            delay_init_mode="normal(0.5, 0.5)",
            behaviour={
                3: ApicalInput(),
                4: BasalInput(),
                5: ProximalInput(),
                6: STDP(a_plus=0.005, a_minus=0.002),
                7: WeightClip(w_min=0.0, w_max=0.5),
            },
        )
        self.exc_inh_conn = SparseSynapseGroup(
            self.exc_pop,
            self.inh_pop,
            net,
            exc_inh_rate,
            weight_init_mode="normal(0.5, 0.5)",
            delay_init_mode="normal(0.5, 0.5)",
            behaviour={
                3: ApicalInput(),
                4: BasalInput(),
                5: ProximalInput(),
                6: STDP(a_plus=0.05, a_minus=0.02),
                7: WeightClip(w_min=0.0, w_max=1.0),
            },
        )
        self.inh_exc_conn = SparseSynapseGroup(
            self.inh_pop,
            self.exc_pop,
            net,
            inh_exc_rate,
            weight_init_mode="normal(0.5, 0.5)",
            delay_init_mode="normal(0.5, 0.5)",
            behaviour={
                3: ApicalInput(),
                4: BasalInput(),
                5: ProximalInput(),
                6: STDP(a_plus=0.02, a_minus=0.01),
                7: WeightClip(w_min=0.0, w_max=0.5),
            },
        )
        self.inh_inh_conn = SparseSynapseGroup(
            self.inh_pop,
            self.inh_pop,
            net,
            inh_inh_rate,
            weight_init_mode="normal(0.25, 0.25)",
            delay_init_mode="normal(0.5, 0.5)",
            behaviour={
                3: ApicalInput(),
                4: BasalInput(),
                5: ProximalInput(),
                6: STDP(a_plus=0.002, a_minus=0.001),
                7: WeightClip(w_min=0.0, w_max=0.5),
            },
        )

    def forward(self, input):
        pass

    def backward(self, delta):
        pass

    def update(self, learning_rate):
        pass


# TODO: Should L5 and L6 be combined?
class L5:
    def __init__(
        self,
        exc_shape,
        inh_shape,
        net,
        exc_exc_rate=0.2,
        exc_inh_rate=0.4,
        inh_exc_rate=0.4,
        inh_inh_rate=0.1,
    ):
        self.exc_pop = SpikingNeuronGroup(
            exc_shape,
            behaviour={
                3: LIF(tau=20, r=1, delta=2, theta_rh=15, alpha=1, beta=1.5),
                4: SpikeTrace(),
                5: SpikeHistory(),
            },
            net=net,
            v_rest=0.0,
            v_reset=-2.0,
            threshold=20,
            tau_s=40.0,
        )
        self.inh_pop = SpikingNeuronGroup(
            inh_shape,
            behaviour={
                3: LIF(tau=20, r=1, delta=2, theta_rh=6, alpha=1, beta=1.5),
                4: SpikeTrace(),
                5: SpikeHistory(),
            },
            net=net,
            v_rest=0.0,
            v_reset=-2.0,
            threshold=10,
            tau_s=30.0,
        )

        self.exc_exc_conn = SparseSynapseGroup(
            self.exc_pop,
            self.exc_pop,
            net,
            exc_exc_rate,
            weight_init_mode="normal(0.25, 0.25)",
            delay_init_mode="normal(0.5, 0.5)",
            behaviour={
                3: ApicalInput(),
                4: BasalInput(),
                5: ProximalInput(),
                6: STDP(a_plus=0.005, a_minus=0.002),
                7: WeightClip(w_min=0.0, w_max=0.5),
            },
        )
        self.exc_inh_conn = SparseSynapseGroup(
            self.exc_pop,
            self.inh_pop,
            net,
            exc_inh_rate,
            weight_init_mode="normal(0.5, 0.5)",
            delay_init_mode="normal(0.5, 0.5)",
            behaviour={
                3: ApicalInput(),
                4: BasalInput(),
                5: ProximalInput(),
                6: STDP(a_plus=0.05, a_minus=0.02),
                7: WeightClip(w_min=0.0, w_max=1.0),
            },
        )
        self.inh_exc_conn = SparseSynapseGroup(
            self.inh_pop,
            self.exc_pop,
            net,
            inh_exc_rate,
            weight_init_mode="normal(0.5, 0.5)",
            delay_init_mode="normal(0.5, 0.5)",
            behaviour={
                3: ApicalInput(),
                4: BasalInput(),
                5: ProximalInput(),
                6: STDP(a_plus=0.02, a_minus=0.01),
                7: WeightClip(w_min=0.0, w_max=0.5),
            },
        )
        self.inh_inh_conn = SparseSynapseGroup(
            self.inh_pop,
            self.inh_pop,
            net,
            inh_inh_rate,
            weight_init_mode="normal(0.25, 0.25)",
            delay_init_mode="normal(0.5, 0.5)",
            behaviour={
                3: ApicalInput(),
                4: BasalInput(),
                5: ProximalInput(),
                6: STDP(a_plus=0.002, a_minus=0.001),
                7: WeightClip(w_min=0.0, w_max=0.5),
            },
        )

    def forward(self, input):
        pass

    def backward(self, delta):
        pass

    def update(self, learning_rate):
        pass


class L6:
    def __init__(
        self,
        exc_shape,
        inh_shape,
        net,
        exc_exc_rate=0.2,
        exc_inh_rate=0.4,
        inh_exc_rate=0.4,
        inh_inh_rate=0.1,
    ):
        self.exc_pop = SpikingNeuronGroup(
            exc_shape,
            behaviour={
                3: LIF(tau=20, r=1, delta=2, theta_rh=15, alpha=1, beta=1.5),
                4: SpikeTrace(),
                5: SpikeHistory(),
            },
            net=net,
            v_rest=0.0,
            v_reset=-2.0,
            threshold=20,
            tau_s=40.0,
        )
        self.inh_pop = SpikingNeuronGroup(
            inh_shape,
            behaviour={
                3: LIF(tau=20, r=1, delta=2, theta_rh=6, alpha=1, beta=1.5),
                4: SpikeTrace(),
                5: SpikeHistory(),
            },
            net=net,
            v_rest=0.0,
            v_reset=-2.0,
            threshold=10,
            tau_s=30.0,
        )

        self.exc_exc_conn = SparseSynapseGroup(
            self.exc_pop,
            self.exc_pop,
            net,
            exc_exc_rate,
            weight_init_mode="normal(0.25, 0.25)",
            delay_init_mode="normal(0.5, 0.5)",
            behaviour={
                3: ApicalInput(),
                4: BasalInput(),
                5: ProximalInput(),
                6: STDP(a_plus=0.005, a_minus=0.002),
                7: WeightClip(w_min=0.0, w_max=0.5),
            },
        )
        self.exc_inh_conn = SparseSynapseGroup(
            self.exc_pop,
            self.inh_pop,
            net,
            exc_inh_rate,
            weight_init_mode="normal(0.5, 0.5)",
            delay_init_mode="normal(0.5, 0.5)",
            behaviour={
                3: ApicalInput(),
                4: BasalInput(),
                5: ProximalInput(),
                6: STDP(a_plus=0.05, a_minus=0.02),
                7: WeightClip(w_min=0.0, w_max=1.0),
            },
        )
        self.inh_exc_conn = SparseSynapseGroup(
            self.inh_pop,
            self.exc_pop,
            net,
            inh_exc_rate,
            weight_init_mode="normal(0.5, 0.5)",
            delay_init_mode="normal(0.5, 0.5)",
            behaviour={
                3: ApicalInput(),
                4: BasalInput(),
                5: ProximalInput(),
                6: STDP(a_plus=0.02, a_minus=0.01),
                7: WeightClip(w_min=0.0, w_max=0.5),
            },
        )
        self.inh_inh_conn = SparseSynapseGroup(
            self.inh_pop,
            self.inh_pop,
            net,
            inh_inh_rate,
            weight_init_mode="normal(0.25, 0.25)",
            delay_init_mode="normal(0.5, 0.5)",
            behaviour={
                3: ApicalInput(),
                4: BasalInput(),
                5: ProximalInput(),
                6: STDP(a_plus=0.002, a_minus=0.001),
                7: WeightClip(w_min=0.0, w_max=0.5),
            },
        )

    def forward(self, input):
        pass

    def backward(self, delta):
        pass

    def update(self, learning_rate):
        pass

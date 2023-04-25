"""
Module of input and output neuronal populations.
"""

from pymonntorch import NeuronGroup, TaggableObject


class InputLayer(TaggableObject):
    def __init__(self):
        self.sensory_neurons = self.SensoryNeurons()
        self.location_neurons = self.LocationNeurons()

    def connect(self, cortical_column, config={}):
        pass

    class SensoryNeurons(NeuronGroup):
        def __init__(self):
            pass

    class LocationNeurons(NeuronGroup):
        def __init__(self):
            pass


class OutputLayer(TaggableObject):
    def __init__(self):
        self.representation_neurons = self.RepresentationNeurons()
        self.motor_neurons = self.MotorNeurons()

    class RepresentationNeurons(NeuronGroup):
        def __init__(self):
            pass

    class MotorNeurons(NeuronGroup):
        def __init__(self):
            pass

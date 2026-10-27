"""
Author: Soroush Mohammaddeimi <soroushdeimi@gmail.com>


Predictive Coding Hierarchy for CoNeX.

Provides high-level structures for building complete predictive
coding networks with multiple hierarchical levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import torch
from torch import Tensor
from pymonntorch import Network, Behavior, NeuronGroup, SynapseGroup

from ..layer import Layer
from ....behaviors.neurons.numenta.predictive_coding import (
    PredictiveCodingConfig,
    PredictionUnit,
    ErrorUnit,
    PrecisionWeighting,
    PredictiveCodingLearning,
    FreeEnergyMinimization,
    create_predictive_layer,
)
from ....behaviors.synapses.numenta.predictive import (
    PredictiveSynapseInit,
    FeedbackPredictionSynapse,
    FeedforwardErrorSynapse,
    PredictiveSynapseLearning,
)


def build_layer(
    network: Network,
    size: int,
    behavior: Dict[int, Behavior],
    tag: str,
) -> Tuple[NeuronGroup, Layer]:
    """Create a population carrying `behavior` and wrap it in a Layer.

    Layer holds neuron groups rather than a size, so the population has to be
    built first. The behaviors live on the population, which is what the
    accessors below read from.

    A predictive coding level is a single population, not an excitatory and
    inhibitory pair, so this uses Layer rather than CorticalLayer.

    Args:
        network: The network both objects belong to.
        size: Number of neurons in the population.
        behavior: Priority-keyed behaviors for the population.
        tag: Tag for the layer; the population gets `<tag>_pop`.

    Returns:
        The population and the layer wrapping it.
    """
    population = NeuronGroup(
        net=network,
        size=size,
        behavior=behavior,
        tag=f"{tag}_pop",
    )
    layer = Layer(
        net=network,
        neurongroups=[population],
        tag=tag,
    )
    return population, layer


@dataclass
class HierarchyLevel:
    """Represents one level in the predictive coding hierarchy.

    Attributes:
        size: Number of neurons at this level.
        name: Name identifier for this level.
        layer: The Layer wrapping this level's population.
        population: The NeuronGroup carrying this level's behaviors.
        config: Predictive coding config for this level.
    """

    size: int
    name: str = ""
    layer: Optional[Layer] = None
    population: Optional[NeuronGroup] = None
    config: Optional[PredictiveCodingConfig] = None


class PredictiveHierarchy:
    """Multi-level predictive coding hierarchy.
    
    Creates a complete predictive coding network with:
    - Multiple hierarchical levels (sensory → abstract)
    - Bidirectional connections (feedforward errors, feedback predictions)
    - Precision weighting at each level
    - Free energy tracking
    
    Based on:
    - Rao & Ballard (1999): Predictive coding in visual cortex
    - Bastos et al. (2012): Canonical microcircuits for predictive coding
    
    Example:
        >>> net = Network()
        >>> hierarchy = PredictiveHierarchy(
        ...     net,
        ...     level_sizes=[784, 256, 64, 16],
        ...     level_names=["sensory", "L4", "L2/3", "abstract"]
        ... )
        >>> hierarchy.build()
        >>> # Set sensory input
        >>> hierarchy.set_input(sensory_data)
        >>> net.simulate(100)
        >>> # Get free energy
        >>> print(hierarchy.get_total_free_energy())
    
    Args:
        network: PyMonntorch Network.
        level_sizes: List of sizes for each level (bottom to top).
        level_names: Optional names for each level.
        config: Base PredictiveCodingConfig.
        build_immediately: Whether to build hierarchy on construction.
    """
    
    def __init__(
        self,
        network: Network,
        level_sizes: List[int],
        level_names: Optional[List[str]] = None,
        config: Optional[PredictiveCodingConfig] = None,
        build_immediately: bool = True,
    ):
        self.network = network
        self.config = config or PredictiveCodingConfig()
        self.n_levels = len(level_sizes)
        
        # Create level specs
        if level_names is None:
            level_names = [f"level_{i}" for i in range(self.n_levels)]
        
        self.levels: List[HierarchyLevel] = [
            HierarchyLevel(size=size, name=name, config=self.config)
            for size, name in zip(level_sizes, level_names)
        ]
        
        # Connection storage
        self.feedforward_synapses: List[SynapseGroup] = []
        self.feedback_synapses: List[SynapseGroup] = []
        
        # Build if requested
        if build_immediately:
            self.build()
    
    def build(self) -> "PredictiveHierarchy":
        """Build the complete predictive hierarchy.
        
        Returns:
            Self for chaining.
        """
        # Create layers
        for level in self.levels:
            behaviors = self._create_level_behaviors(level)
            level.population, level.layer = build_layer(
                self.network, level.size, behaviors, level.name
            )
        
        # Create connections
        for i in range(self.n_levels - 1):
            lower = self.levels[i]
            higher = self.levels[i + 1]
            
            # Feedforward (error) connection: lower → higher
            ff_syn = self._create_feedforward_synapse(lower.population, higher.population)
            self.feedforward_synapses.append(ff_syn)

            # Feedback (prediction) connection: higher → lower
            fb_syn = self._create_feedback_synapse(higher.population, lower.population)
            self.feedback_synapses.append(fb_syn)
        
        return self
    
    def _create_level_behaviors(self, level: HierarchyLevel) -> Dict[int, Behavior]:
        """Create behaviors for a hierarchy level."""
        cfg = level.config or self.config
        
        return {
            100: PredictionUnit(config=cfg),
            110: ErrorUnit(config=cfg, precision_weighted=True),
            120: PrecisionWeighting(config=cfg, learnable=True),
            130: PredictiveCodingLearning(config=cfg),
            140: FreeEnergyMinimization(config=cfg, track_components=True),
        }
    
    def _create_feedforward_synapse(
        self,
        src: NeuronGroup,
        dst: NeuronGroup,
    ) -> SynapseGroup:
        """Create feedforward (error propagation) synapse.

        The predictive synapse behaviors read src.size and src.activity, so
        they connect populations directly rather than sitting on a Synapsis.
        """
        behaviors = {
            100: PredictiveSynapseInit(
                config=self.config, connection_type="feedforward"
            ),
            110: FeedforwardErrorSynapse(
                config=self.config, precision_weighted=True
            ),
            120: PredictiveSynapseLearning(
                config=self.config, learning_type="error_driven"
            ),
        }

        return SynapseGroup(
            net=self.network,
            src=src,
            dst=dst,
            behavior=behaviors,
            tag=f"ff_{src.tags[0]}_{dst.tags[0]}",
        )
    
    def _create_feedback_synapse(
        self,
        src: NeuronGroup,
        dst: NeuronGroup,
    ) -> SynapseGroup:
        """Create feedback (prediction) synapse."""
        behaviors = {
            100: PredictiveSynapseInit(
                config=self.config, connection_type="feedback"
            ),
            110: FeedbackPredictionSynapse(config=self.config),
            120: PredictiveSynapseLearning(
                config=self.config, learning_type="error_driven"
            ),
        }

        return SynapseGroup(
            net=self.network,
            src=src,
            dst=dst,
            behavior=behaviors,
            tag=f"fb_{src.tags[0]}_{dst.tags[0]}",
        )
    
    def set_input(self, data: Tensor) -> None:
        """Set sensory input to the bottom level.
        
        Args:
            data: Input tensor matching bottom level size.
        """
        bottom = self.levels[0].population
        if bottom is not None:
            bottom.activity = data.to(
                device=self.network.device,
                dtype=self.network.def_dtype,
            )
    
    def get_predictions(self, level: int = 0) -> Tensor:
        """Get predictions at a specific level.
        
        Args:
            level: Level index (0 = bottom/sensory).
        
        Returns:
            Prediction tensor.
        """
        population = self.levels[level].population
        if population is not None and hasattr(population, "prediction"):
            return population.prediction
        return torch.zeros(self.levels[level].size)
    
    def get_errors(self, level: int = 0) -> Tensor:
        """Get prediction errors at a specific level.
        
        Args:
            level: Level index.
        
        Returns:
            Error tensor.
        """
        population = self.levels[level].population
        if population is not None and hasattr(population, "prediction_error"):
            return population.prediction_error
        return torch.zeros(self.levels[level].size)
    
    def get_free_energy(self, level: int = 0) -> float:
        """Get free energy at a specific level.
        
        Args:
            level: Level index.
        
        Returns:
            Free energy value.
        """
        population = self.levels[level].population
        if population is not None and hasattr(population, "free_energy"):
            return population.free_energy.item()
        return 0.0
    
    def get_total_free_energy(self) -> float:
        """Get total free energy across all levels.
        
        Returns:
            Sum of free energy at all levels.
        """
        return sum(self.get_free_energy(i) for i in range(self.n_levels))
    
    def get_precision(self, level: int = 0) -> Tensor:
        """Get precision estimates at a specific level.
        
        Args:
            level: Level index.
        
        Returns:
            Precision tensor.
        """
        population = self.levels[level].population
        if population is not None and hasattr(population, "precision"):
            return population.precision
        return torch.ones(self.levels[level].size)
    
    def __repr__(self) -> str:
        """String representation of hierarchy."""
        level_strs = [f"{l.name}({l.size})" for l in self.levels]
        return f"PredictiveHierarchy[{' → '.join(level_strs)}]"


class PredictiveCorticalColumn:
    """Cortical column with predictive coding microcircuit.
    
    Implements the canonical microcircuit for predictive coding
    as described by Bastos et al. (2012):
    
    - L4: Receives feedforward input, computes errors
    - L2/3: Generates predictions, sends feedback
    - L5: Output layer, motor predictions
    - L6: Modulates precision
    
    Args:
        network: PyMonntorch Network.
        column_size: Base size for column layers.
        config: PredictiveCodingConfig.
        name: Column identifier.
    """
    
    def __init__(
        self,
        network: Network,
        column_size: int = 100,
        config: Optional[PredictiveCodingConfig] = None,
        name: str = "column",
    ):
        self.network = network
        self.column_size = column_size
        self.config = config or PredictiveCodingConfig()
        self.name = name
        
        # Layer sizes based on cortical proportions
        self.layer_sizes = {
            "L4": column_size,  # Granular layer (input)
            "L23": int(column_size * 1.5),  # Supragranular (prediction)
            "L5": int(column_size * 0.8),  # Infragranular (output)
            "L6": int(column_size * 0.5),  # Precision modulation
        }
        
        # Layers, and the populations that carry their behaviors
        self.L4: Optional[Layer] = None
        self.L23: Optional[Layer] = None
        self.L5: Optional[Layer] = None
        self.L6: Optional[Layer] = None
        self.populations: Dict[str, NeuronGroup] = {}

        # Connections
        self.synapses: Dict[str, SynapseGroup] = {}
        
        # Build column
        self._build()
    
    def _build(self) -> None:
        """Build the cortical column microcircuit."""
        cfg = self.config
        
        # L4: Error computation layer (receives FF input)
        self.populations["L4"], self.L4 = build_layer(
            self.network,
            self.layer_sizes["L4"],
            {
                100: ErrorUnit(config=cfg, precision_weighted=True),
                110: PrecisionWeighting(config=cfg),
                120: FreeEnergyMinimization(config=cfg),
            },
            f"{self.name}_L4",
        )

        # L2/3: Prediction generation layer
        self.populations["L23"], self.L23 = build_layer(
            self.network,
            self.layer_sizes["L23"],
            {
                100: PredictionUnit(config=cfg, use_temporal=True),
                110: ErrorUnit(config=cfg),
                120: PredictiveCodingLearning(config=cfg),
            },
            f"{self.name}_L23",
        )

        # L5: Output/motor prediction layer
        self.populations["L5"], self.L5 = build_layer(
            self.network,
            self.layer_sizes["L5"],
            {
                100: PredictionUnit(config=cfg),
                110: PredictiveCodingLearning(config=cfg),
            },
            f"{self.name}_L5",
        )

        # L6: Precision modulation layer
        self.populations["L6"], self.L6 = build_layer(
            self.network,
            self.layer_sizes["L6"],
            {
                100: PrecisionWeighting(config=cfg, learnable=True),
            },
            f"{self.name}_L6",
        )

        # Create canonical connections
        self._create_connections()
    
    def _create_connections(self) -> None:
        """Create canonical microcircuit connections.

        The predictive synapse behaviors read src.size and src.activity, so
        these connect the layer populations rather than the Layer containers.
        """
        cfg = self.config
        pop = self.populations

        def connect(src: str, dst: str, behavior: Dict[int, Behavior]) -> SynapseGroup:
            return SynapseGroup(
                net=self.network,
                src=pop[src],
                dst=pop[dst],
                behavior=behavior,
                tag=f"{self.name}_{src}_{dst}",
            )

        # L4 → L2/3: Feedforward (error propagation)
        self.synapses["L4_to_L23"] = connect("L4", "L23", {
            100: PredictiveSynapseInit(config=cfg, connection_type="feedforward"),
            110: FeedforwardErrorSynapse(config=cfg),
            120: PredictiveSynapseLearning(config=cfg),
        })

        # L2/3 → L4: Feedback (predictions)
        self.synapses["L23_to_L4"] = connect("L23", "L4", {
            100: PredictiveSynapseInit(config=cfg, connection_type="feedback"),
            110: FeedbackPredictionSynapse(config=cfg),
            120: PredictiveSynapseLearning(config=cfg),
        })

        # L2/3 → L5: Output generation
        self.synapses["L23_to_L5"] = connect("L23", "L5", {
            100: PredictiveSynapseInit(config=cfg, connection_type="feedforward"),
            110: FeedforwardErrorSynapse(config=cfg),
        })

        # L6 → L4: Precision modulation
        self.synapses["L6_to_L4"] = connect("L6", "L4", {
            100: PredictiveSynapseInit(config=cfg, connection_type="lateral"),
        })
    
    def set_input(self, data: Tensor) -> None:
        """Set feedforward input to L4.
        
        Args:
            data: Input tensor.
        """
        if "L4" in self.populations:
            self.populations["L4"].activity = data.to(
                device=self.network.device,
                dtype=self.network.def_dtype,
            )
    
    def get_prediction(self) -> Tensor:
        """Get current prediction from L2/3."""
        l23 = self.populations.get("L23")
        if l23 is not None and hasattr(l23, "prediction"):
            return l23.prediction
        return torch.zeros(self.layer_sizes["L23"])
    
    def get_output(self) -> Tensor:
        """Get output from L5."""
        l5 = self.populations.get("L5")
        if l5 is not None and hasattr(l5, "activity"):
            return l5.activity
        return torch.zeros(self.layer_sizes["L5"])
    
    def get_error(self) -> Tensor:
        """Get prediction error from L4."""
        l4 = self.populations.get("L4")
        if l4 is not None and hasattr(l4, "prediction_error"):
            return l4.prediction_error
        return torch.zeros(self.layer_sizes["L4"])
    
    def get_free_energy(self) -> float:
        """Get free energy of the column."""
        if self.L4 is not None and hasattr(self.L4, "free_energy"):
            return self.L4.free_energy.item()
        return 0.0
    
    def __repr__(self) -> str:
        return f"PredictiveCorticalColumn('{self.name}', size={self.column_size})"

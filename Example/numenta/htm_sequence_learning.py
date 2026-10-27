"""HTM sequence learning and anomaly detection.

Runs the classic Numenta pipeline end to end:

    symbol -> SDR -> Spatial Pooler -> Temporal Memory -> anomaly score

A repeating sequence is presented for several epochs. Temporal Memory learns
which element follows which, so its cells become predictive and the anomaly
score falls to zero.

An unexpected symbol is then injected into the learned sequence. The anomaly
score spikes on that symbol, and again on the one after it: the surprise breaks
the context, so the next element has no prediction to match either. Columns with
no predicted cell burst, which is what the bursting count tracks.

Run:
    python Example/numenta/htm_sequence_learning.py
"""

import torch
from pymonntorch import Network, NeuronGroup

from conex import (
    SDR,
    SpatialPooler,
    SpatialPoolerConfig,
    TemporalMemory,
    TemporalMemoryConfig,
    TimeResolution,
)


INPUT_SIZE = 256
N_COLUMNS = 128
CELLS_PER_COLUMN = 8
SEQUENCE = ["A", "B", "C", "D"]
EPOCHS = 12
SEED = 1234


def encode_symbols(symbols, input_size, n_active, seed):
    """Give every symbol a fixed random SDR, so inputs are sparse and distinct."""
    generator = torch.Generator().manual_seed(seed)
    return {
        symbol: SDR(
            size=input_size,
            indices=torch.randperm(input_size, generator=generator)[:n_active],
        )
        for symbol in symbols
    }


def build_network():
    net = Network(behavior={1: TimeResolution(dt=1.0)}, dtype=torch.float32, device="cpu")

    pooler_config = SpatialPoolerConfig(
        input_size=INPUT_SIZE,
        n_columns=N_COLUMNS,
        potential_pct=0.6,
        local_area_density=0.08,
        syn_perm_connected=0.1,
        syn_perm_active_inc=0.06,
        syn_perm_inactive_dec=0.01,
        seed=SEED,
    )
    pooler = NeuronGroup(
        net=net,
        size=pooler_config.n_columns,
        behavior={100: SpatialPooler(config=pooler_config)},
        tag="pooler",
    )

    memory_config = TemporalMemoryConfig(
        n_columns=N_COLUMNS,
        cells_per_column=CELLS_PER_COLUMN,
        activation_threshold=3,
        learning_threshold=2,
        max_new_synapses=12,
        permanence_increment=0.12,
        permanence_decrement=0.04,
    )
    memory = NeuronGroup(
        net=net,
        size=memory_config.n_columns * memory_config.cells_per_column,
        behavior={110: TemporalMemory(config=memory_config)},
        tag="memory",
    )

    net.initialize()
    return net, pooler, memory


def step(pooler, memory, pattern, learn=True):
    """Push one symbol through the pooler and into temporal memory."""
    active_columns = pooler.spatial_pooler.compute(pooler, pattern, learn=learn)
    memory.temporal_memory.compute(memory, active_columns, learn=learn)
    return {
        "active_columns": int(active_columns.sum()),
        "bursting_columns": int(memory.tm_bursting_columns.sum()),
        "anomaly": float(memory.tm_anomaly_score),
    }


def main():
    torch.manual_seed(SEED)

    codes = encode_symbols(
        SEQUENCE + ["X"], INPUT_SIZE, n_active=INPUT_SIZE // 20, seed=SEED
    )
    net, pooler, memory = build_network()

    print(f"input {INPUT_SIZE} bits -> {N_COLUMNS} columns x {CELLS_PER_COLUMN} cells")
    print(f"sequence {' '.join(SEQUENCE)} for {EPOCHS} epochs\n")

    print("epoch  mean anomaly  bursting columns")
    for epoch in range(EPOCHS):
        anomalies, bursts = [], []
        for symbol in SEQUENCE:
            result = step(pooler, memory, codes[symbol].to_float())
            anomalies.append(result["anomaly"])
            bursts.append(result["bursting_columns"])
        mean_anomaly = sum(anomalies) / len(anomalies)
        mean_burst = sum(bursts) / len(bursts)
        print(f"{epoch:>5}  {mean_anomaly:>12.3f}  {mean_burst:>16.1f}")

    print("\nNow break the pattern: A B X D")
    print("symbol  anomaly  bursting columns")
    for symbol in ["A", "B", "X", "D"]:
        result = step(pooler, memory, codes[symbol].to_float(), learn=False)
        print(f"{symbol:>6}  {result['anomaly']:>7.3f}  {result['bursting_columns']:>16}")

    print("\nSDR overlap between symbol codes (semantic distance):")
    for a in SEQUENCE:
        row = [f"{codes[a].overlap(codes[b]):>3}" for b in SEQUENCE]
        print(f"  {a}  {' '.join(row)}")


if __name__ == "__main__":
    main()

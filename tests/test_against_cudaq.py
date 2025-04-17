import numpy as np
import cudaq
from typing import List
from qaoa_fp16.qaoa_algorithm import MaxCutQAOA_FP16
from qaoa_fp16.utils import fidelity
import logging
import matplotlib.pyplot as plt
from qaoa_fp16.utils import fidelity, get_norm_deviation

# pytest tests -v -s --log-cli-level info

@cudaq.kernel
def kernel_qaoa(qubit_count: int, layer_count: int, edges_src: List[int],
                edges_tgt: List[int], thetas: List[float]):
    """Build the QAOA circuit for max cut of the graph with given edges and nodes
    
    Args:
        qubit_count: Number of qubits in the circuit, which is the same as the number of nodes
        layer_count: Number of layers in the QAOA kernel
        edges_src: List of the first (source) node in each edge
        edges_tgt: List of the second (target) node in each edge
        thetas: Free variables to be optimized
    """
    # Let's allocate the qubits
    qreg = cudaq.qvector(qubit_count)
    # And then place the qubits in superposition
    h(qreg)

    # Each layer has two components: the problem kernel and the mixer
    for i in range(layer_count):
        # Add the problem kernel to each layer
        for edge in range(len(edges_src)):
            qubitu = edges_src[edge]
            qubitv = edges_tgt[edge]
            # Apply the problem Hamiltonian for this edge
            x.ctrl(qreg[qubitu], qreg[qubitv])
            rz(2.0 * thetas[i], qreg[qubitv])
            x.ctrl(qreg[qubitu], qreg[qubitv])
            
        # Add the mixer kernel to each layer
        for j in range(qubit_count):
            rx(2.0 * thetas[i + layer_count], qreg[j])

def test_graph_circuit(caplog):
    """
    Tests a given maxcut QAOA circuit against CUDA-Q
    Applies the circuit a single time to an initial state and compares the state.
    """
    # caplog.set_level(logging.INFO)

    # We'll use the graph below to illustrate how QAOA can be used to
    # solve a max cut problem

    #       v1  0--------------0 v2
    #           |              | \
    #           |              |  \
    #           |              |   \
    #           |              |    \
    #       v0  0--------------0 v3-- 0 v4
    # The max cut solutions are 01011, 10100, 01010, 10101 .

    # First we define the graph nodes (i.e., vertices) and edges as lists of integers so that they can be broadcast into
    # a cudaq.kernel.
    nodes: List[int] = [0, 1, 2, 3, 4]
    edges = [[0, 1], [1, 2], [2, 3], [3, 0], [2, 4], [3, 4]]
    edges_src: List[int] = [edges[i][0] for i in range(len(edges))]
    edges_tgt: List[int] = [edges[i][1] for i in range(len(edges))]

    # Create the FP16 QAOA solver
    qaoa_fp16 = MaxCutQAOA_FP16(nodes, edges, layer_count=2, seed=42)
    logging.info(f"Number of nodes: {len(qaoa_fp16.nodes)}")
    logging.info(f"Number of edges: {len(qaoa_fp16.edges)}")
    logging.info(f"Number of qubits: {qaoa_fp16.qubit_count}")
    logging.info(f"Number of parameters: {qaoa_fp16.parameter_count}")
    logging.info(f"Number of layers: {qaoa_fp16.layer_count}")
    
    initial_params = np.random.uniform(-np.pi / 8, np.pi / 8, qaoa_fp16.parameter_count)
    logging.info(f"Initial parameters: {initial_params}")

    # TODO: Compare application of fp16 circuit on initial params against cudaq
    fp16_state = qaoa_fp16.apply_qaoa_circuit(initial_params.tolist())
    fp16_state = [complex(x.real, x.imag) for x in fp16_state]
    cudaq_state = cudaq.get_state(kernel_qaoa, qaoa_fp16.qubit_count, qaoa_fp16.layer_count, edges_src, edges_tgt, initial_params.tolist())

    fp16_norm_deviation = get_norm_deviation(fp16_state)
    cudaq_norm_deviation = get_norm_deviation(cudaq_state)
    logging.info(f"FP16 norm deviation: {fp16_norm_deviation}")
    logging.info(f"CUDA-Q norm deviation: {cudaq_norm_deviation}")

    mismatch = False 
    max_diff = -1
    total_diff = 0
    for u, v in zip(fp16_state, cudaq_state):
        diff = abs(u - v)
        logging.info(f"FP16: {u}, CUDA-Q: {v}, Diff: {diff}")
        total_diff += diff
        if diff > max_diff:
            max_diff = diff

    fid = fidelity(fp16_state, cudaq_state)
    logging.info(f"Fidelity: {fid}")
    logging.info(f"Max diff: {max_diff}")
    logging.info(f"Average diff: {total_diff / len(fp16_state)}")

    assert not mismatch, "Found mismatches between FP16 and CUDA-Q states after applying QAOA circuit!"

def plot_state_counts(counts, filename):
    # Calculate probabilities
    total = sum(counts.values())
    probs = {state: count / total for state, count in counts.items()}

    states = sorted(probs.keys())
    probabilities = [probs[state] for state in states]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot the bar chart
    bars = ax.bar(states, probabilities)
    
    # Add title and labels
    ax.set_title('Probability Distribution of States', fontsize=14)
    ax.set_xlabel('State', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    
    # Rotate x-axis labels for better readability if there are many states
    if len(states) > 10:
        plt.xticks(rotation=45, ha='right')
    
    # Add a grid for easier reading
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of each bar
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.4f}',
                ha='center', va='bottom', rotation=0)
    
    # Adjust layout to fit all elements
    plt.tight_layout()
    
    # Show the plot
    plt.savefig(f"{filename}.png")
    
    # Return the data for further use if needed
    return states, probabilities

def test_graph_sample(caplog):
    nodes: List[int] = [0, 1, 2, 3, 4]
    edges = [[0, 1], [1, 2], [2, 3], [3, 0], [2, 4], [3, 4]]
    edges_src: List[int] = [edges[i][0] for i in range(len(edges))]
    edges_tgt: List[int] = [edges[i][1] for i in range(len(edges))]

    qaoa_fp16 = MaxCutQAOA_FP16(nodes, edges, layer_count=2, seed=42)
    initial_params = np.random.uniform(-np.pi / 8, np.pi / 8, qaoa_fp16.parameter_count)

    cudaq_counts = cudaq.sample(kernel_qaoa, qaoa_fp16.qubit_count, qaoa_fp16.layer_count, edges_src, edges_tgt, initial_params.tolist(), shots_count=1000)
    plot_state_counts(cudaq_counts, "cudaq_state_counts")


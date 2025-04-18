import numpy as np
import cudaq
from cudaq import spin
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

def cudaq_build_maxcut_hamiltonian(edges_src, edges_tgt):
    """Create the Hamiltonian for finding the max cut for the graph
    
    Returns:
        Hamiltonian for finding the max cut of the graph
    """
    print(edges_src)
    print(edges_tgt)
    hamiltonian = 0
    for edge in range(len(edges_src)):
        qubitu = edges_src[edge]
        qubitv = edges_tgt[edge]
        # Add a term to the Hamiltonian for the edge (u,v)
        hamiltonian += 0.5 * (spin.z(qubitu) * spin.z(qubitv) - 
                             spin.i(qubitu) * spin.i(qubitv))
    return hamiltonian

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
    
    # Adjust layout to fit all elements
    plt.tight_layout()
    
    # Show the plot
    plt.savefig(f"{filename}.png")
    
    # Return the data for further use if needed
    return states, probabilities

def plot_state_counts_comparison(counts1, counts2, filename, label1="Distribution 1", label2="Distribution 2"):
    """Plot two count distributions side by side for comparison
    
    Args:
        counts1: First count distribution dictionary
        counts2: Second count distribution dictionary 
        filename: Name for saving the plot
        label1: Label for first distribution
        label2: Label for second distribution
    """
    # Calculate probabilities
    total1 = sum(counts1.values())
    total2 = sum(counts2.values())
    probs1 = {state: count/total1 for state, count in counts1.items()}
    probs2 = {state: count/total2 for state, count in counts2.items()}
    
    # Get all unique states
    all_states = sorted(set(list(probs1.keys()) + list(probs2.keys())))
    
    # Fill in missing states with 0 probability
    prob_list1 = [probs1.get(state, 0) for state in all_states]
    prob_list2 = [probs2.get(state, 0) for state in all_states]
    
    # Convert to numpy arrays for plotting
    x = np.arange(len(all_states))
    width = 0.35
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Create bars
    rects1 = ax.bar(x - width/2, prob_list1, width, label=label1)
    rects2 = ax.bar(x + width/2, prob_list2, width, label=label2)
    
    # Add labels and title
    ax.set_title('Comparison of State Distributions', fontsize=14)
    ax.set_xlabel('State', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(all_states)
    
    # Rotate x-axis labels if needed
    if len(all_states) > 10:
        plt.xticks(rotation=45, ha='right')
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f"{filename}.png")
    plt.close()

def total_variation_distance(counts1, counts2):
    """Calculate TVD between two count dictionaries"""
    # Get all possible bitstrings
    bitstrings = [x for x, _ in counts1.items()]
    
    # Convert to probabilities
    total1 = sum(counts1.values())
    total2 = sum(counts2.values())
    
    prob1 = {k: counts1[k] / total1 for k in bitstrings}
    prob2 = {k: counts2[k] / total2 for k in bitstrings}
    
    # Calculate TVD
    tvd = 0.5 * sum(abs(prob1[k] - prob2[k]) for k in bitstrings)
    
    return tvd  # Range: [0, 1], lower is better

def test_graph_sample_random_params(caplog):
    nodes: List[int] = [0, 1, 2, 3, 4]
    edges = [[0, 1], [1, 2], [2, 3], [3, 0], [2, 4], [3, 4]]
    edges_src: List[int] = [edges[i][0] for i in range(len(edges))]
    edges_tgt: List[int] = [edges[i][1] for i in range(len(edges))]

    qaoa_fp16 = MaxCutQAOA_FP16(nodes, edges, layer_count=2, seed=42)
    initial_params = np.random.uniform(-np.pi / 8, np.pi / 8, qaoa_fp16.parameter_count)

    cudaq_counts = cudaq.sample(kernel_qaoa, qaoa_fp16.qubit_count, qaoa_fp16.layer_count, edges_src, edges_tgt, initial_params.tolist(), shots_count=1000)
    fp16_counts = qaoa_fp16.sample(initial_params.tolist(), shots=1000)
    
    # Generate individual distribution plots
    plot_state_counts(cudaq_counts, "cudaq_state_counts")
    plot_state_counts(fp16_counts, "fp16_state_counts")
    
    # Generate comparison plot
    plot_state_counts_comparison(cudaq_counts, fp16_counts, "state_counts_comparison", 
                               label1="CUDA-Q", label2="FP16")
    
    # Calculate TVD
    tvd = total_variation_distance(cudaq_counts, fp16_counts)
    logging.info(f"Total Variation Distance: {tvd}")

def test_graph_eval_expectation_random_params(caplog):
    nodes: List[int] = [0, 1, 2, 3, 4]
    edges = [[0, 1], [1, 2], [2, 3], [3, 0], [2, 4], [3, 4]]
    edges_src: List[int] = [edges[i][0] for i in range(len(edges))]
    edges_tgt: List[int] = [edges[i][1] for i in range(len(edges))]

    qaoa_fp16 = MaxCutQAOA_FP16(nodes, edges, layer_count=2, seed=42)
    logging.info(f"fp16 hamiltonian: \n{qaoa_fp16.hamiltonian}")
    cudaq_hamiltonian = cudaq_build_maxcut_hamiltonian(edges_src, edges_tgt)
    logging.info(f"CUDA-Q Hamiltonian: \n{cudaq_hamiltonian}")

    initial_params = np.random.uniform(-np.pi / 8, np.pi / 8, qaoa_fp16.parameter_count)

    fp16_state = qaoa_fp16.apply_qaoa_circuit(initial_params.tolist())
    fp16_expectation = qaoa_fp16.evaluate_expectation(fp16_state)
    logging.info(f"fp16 expectation: {fp16_expectation}")

    cudaq_state = cudaq.get_state(kernel_qaoa, qaoa_fp16.qubit_count, qaoa_fp16.layer_count, edges_src, edges_tgt, initial_params.tolist())
    cudaq_expectation = cudaq.observe(kernel_qaoa, cudaq_hamiltonian, qaoa_fp16.qubit_count, qaoa_fp16.layer_count,
                            edges_src, edges_tgt, initial_params.tolist()).expectation()
    logging.info(f"CUDA-Q expectation: {cudaq_expectation}")

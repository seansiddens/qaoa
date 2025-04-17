import unittest
import numpy as np
import cudaq
from typing import List
from qaoa_fp16.qaoa_algorithm import MaxCutQAOA_FP16
from qaoa_fp16.utils import fidelity
import logging
import pytest

# pytest tests -v -s --log-cli-level info

def test_graph(caplog):
    caplog.set_level(logging.INFO)

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

    # assert True

#     # Parameters to test
#     gamma = np.pi / 8
#     beta = np.pi / 4
#     parameters = [gamma, beta]
    
#     # Get state from FP16 implementation
#     fp16_state = qaoa_fp16.apply_qaoa_circuit(parameters)
#     fp16_state_complex = np.array([x.to_complex64() for x in fp16_state])
    
#     # Create CUDA-Q kernel
#     @cudaq.kernel
#     def qaoa_kernel(gamma: float, beta: float, edges_src: List[int], edges_tgt: List[int]):
#         qubits = cudaq.qvector(len(nodes))
#         h(qubits)
        
#         for edge_idx in range(len(edges_src)):
#             qubit_u = edges_src[edge_idx]
#             qubit_v = edges_tgt[edge_idx]
#             x.ctrl(qubits[qubit_u], qubits[qubit_v])
#             rz(2.0 * gamma, qubits[qubit_v])
#             x.ctrl(qubits[qubit_u], qubits[qubit_v])
        
#         for qubit_idx in range(len(nodes)):
#             rx(2.0 * beta, qubits[qubit_idx])
    
#     # Extract edges for CUDA-Q
#     edges_src = [edge[0] for edge in edges]
#     edges_tgt = [edge[1] for edge in edges]
    
#     # Get CUDA-Q state
#     cudaq_state = cudaq.get_state(qaoa_kernel, gamma, beta, edges_src, edges_tgt)
    
#     # Compare states
    # fid = fidelity(fp16_state_complex, cudaq_state)
    # assert fid > 0.99  # Allow for small numerical differences


# # Test more complex graphs against CUDA-Q

import numpy as np
from typing import List
from .complex_fp16 import ComplexFP16


def qaoa_problem_kernel_fp16(gamma, edge_list, num_qubits):
    """Implement the problem Hamiltonian part of QAOA using FP16
    
    Args:
        gamma: The problem Hamiltonian angle parameter
        edge_list: List of edges as [source, target] pairs
        num_qubits: Total number of qubits
        
    Returns:
        Matrix representing the unitary exp(-i*gamma*H_problem)
    """
    dim = 2 ** num_qubits
    
    # Start with identity matrix
    result = np.zeros((dim, dim), dtype=object)
    for i in range(dim):
        for j in range(dim):
            if i == j:
                result[i, j] = ComplexFP16(1.0, 0.0)
            else:
                result[i, j] = ComplexFP16(0.0, 0.0)
    
    # For each edge, apply exp(-i*gamma*Z_i Z_j)
    for edge in edge_list:
        i, j = edge
        
        # Create the ZZ interaction matrix for qubits i and j
        zz_unitary = np.zeros((dim, dim), dtype=object)
        for state_idx in range(dim):
            # Convert state index to binary representation
            bin_rep = format(state_idx, f'0{num_qubits}b')
            
            # Check if the qubits i and j have the same or different parity
            if bin_rep[num_qubits - 1 - i] == bin_rep[num_qubits - 1 - j]:
                # Same parity means eigenvalue +1 for Z_i Z_j, so e^(-i*gamma*1)
                phase = -gamma
                zz_unitary[state_idx, state_idx] = ComplexFP16(np.cos(phase), np.sin(phase))
            else:
                # Different parity means eigenvalue -1 for Z_i Z_j, so e^(-i*gamma*(-1))
                phase = gamma
                zz_unitary[state_idx, state_idx] = ComplexFP16(np.cos(phase), np.sin(phase))
        
        # Apply this edge's ZZ interaction by matrix multiplication
        temp = np.zeros((dim, dim), dtype=object)
        for k in range(dim):
            for l in range(dim):
                temp[k, l] = ComplexFP16(0.0, 0.0)
                for m in range(dim):
                    temp[k, l] = temp[k, l] + zz_unitary[k, m] * result[m, l]
        
        # Update the result
        result = temp
    
    return result

def qaoa_mixer_kernel_fp16(beta, num_qubits):
    """Implement the mixer Hamiltonian part of QAOA using FP16
    
    Args:
        beta: The mixer Hamiltonian angle parameter
        num_qubits: Total number of qubits
        
    Returns:
        Matrix representing the unitary exp(-i*beta*H_mixer)
    """
    dim = 2 ** num_qubits
    
    # Start with identity matrix
    result = np.zeros((dim, dim), dtype=object)
    for i in range(dim):
        for j in range(dim):
            if i == j:
                result[i, j] = ComplexFP16(1.0, 0.0)
            else:
                result[i, j] = ComplexFP16(0.0, 0.0)
    
    # For each qubit, apply RX(2*beta)
    for qubit in range(num_qubits):
        # Create the RX matrix for this qubit
        rx_matrix = np.zeros((dim, dim), dtype=object)
        
        for state_idx1 in range(dim):
            for state_idx2 in range(dim):
                # Get binary representations of both states
                bin_rep1 = format(state_idx1, f'0{num_qubits}b')
                bin_rep2 = format(state_idx2, f'0{num_qubits}b')
                
                # Check if states differ only in the target qubit
                differs_only_at_target = True
                for q in range(num_qubits):
                    if q != qubit and bin_rep1[num_qubits - 1 - q] != bin_rep2[num_qubits - 1 - q]:
                        differs_only_at_target = False
                        break
                
                if differs_only_at_target:
                    if bin_rep1[num_qubits - 1 - qubit] == bin_rep2[num_qubits - 1 - qubit]:
                        # Diagonal element - cos(beta)
                        rx_matrix[state_idx1, state_idx2] = ComplexFP16(np.cos(beta), 0.0)
                    else:
                        # Off-diagonal element - -i*sin(beta)
                        rx_matrix[state_idx1, state_idx2] = ComplexFP16(0.0, -np.sin(beta))
                else:
                    # States differ by more than one qubit - no contribution
                    rx_matrix[state_idx1, state_idx2] = ComplexFP16(0.0, 0.0)
        
        # Apply this qubit's RX by matrix multiplication
        temp = np.zeros((dim, dim), dtype=object)
        for k in range(dim):
            for l in range(dim):
                temp[k, l] = ComplexFP16(0.0, 0.0)
                for m in range(dim):
                    temp[k, l] = temp[k, l] + rx_matrix[k, m] * result[m, l]
        
        # Update the result
        result = temp
    
    return result

def qaoa_apply_fp16(initial_state, problem_kernel, mixer_kernel):
    """Apply one layer of QAOA to an initial state using FP16
    
    Args:
        initial_state: Initial state vector
        problem_kernel: Problem Hamiltonian unitary matrix
        mixer_kernel: Mixer Hamiltonian unitary matrix
        
    Returns:
        Resulting state vector
    """
    dim = len(initial_state)
    
    # Apply problem kernel
    intermediate_state = np.zeros(dim, dtype=object)
    for i in range(dim):
        intermediate_state[i] = ComplexFP16(0.0, 0.0)
        for j in range(dim):
            intermediate_state[i] = intermediate_state[i] + problem_kernel[i, j] * initial_state[j]
    
    # Apply mixer kernel
    final_state = np.zeros(dim, dtype=object)
    for i in range(dim):
        final_state[i] = ComplexFP16(0.0, 0.0)
        for j in range(dim):
            final_state[i] = final_state[i] + mixer_kernel[i, j] * intermediate_state[j]
    
    return final_state
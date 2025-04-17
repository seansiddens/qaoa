import numpy as np
from typing import List
from .complex_fp16 import ComplexFP16
from .circuit_components import qaoa_problem_kernel_fp16, qaoa_mixer_kernel_fp16

class MaxCutQAOA_FP16:
    """QAOA implementation for Max-Cut using FP16 complex numbers"""
    def __init__(self, nodes: List[int], edges: List[List[int]], layer_count: int = 1, seed: int = 13):
        """Initialize the QAOA solver
        
        Args:
            nodes: List of nodes in the graph
            edges: List of edges as [source, target] pairs
            layer_count: Number of QAOA layers
            seed: Random seed for reproducibility
        """
        self.nodes = nodes
        self.edges = edges
        self.layer_count = layer_count
        self.seed = seed
        
        self.qubit_count = len(nodes)
        self.parameter_count = 2 * layer_count
        
        # Set random seed
        np.random.seed(seed)
        
        # Create the Hamiltonian
        self.hamiltonian = self.hamiltonian_max_cut()
    
    def hamiltonian_max_cut(self):
        """Create the Hamiltonian for finding the max cut for the graph in FP16 format
        
        Returns:
            Dictionary representation of the Hamiltonian terms and their coefficients
        """
        # For FP16 representation, we'll create a simple representation of the Hamiltonian
        # as a dictionary of Pauli strings and their coefficients
        hamiltonian = {}
        
        for edge in self.edges:
            u, v = edge
            
            # The term Z_u Z_v with coefficient 0.5
            term = ['I'] * self.qubit_count
            term[u] = 'Z'
            term[v] = 'Z'
            pauli_string = ''.join(term)
            
            if pauli_string in hamiltonian:
                hamiltonian[pauli_string] += ComplexFP16(0.5, 0.0)
            else:
                hamiltonian[pauli_string] = ComplexFP16(0.5, 0.0)
            
            # The term -I_u I_v with coefficient -0.5
            identity_term = 'I' * self.qubit_count
            if identity_term in hamiltonian:
                hamiltonian[identity_term] -= ComplexFP16(0.5, 0.0)
            else:
                hamiltonian[identity_term] = ComplexFP16(-0.5, 0.0)
        
        return hamiltonian
    
    def create_initial_state(self):
        """Create the initial |+⟩^⊗n state using FP16
        
        Returns:
            Array representing the initial state
        """
        dim = 2 ** self.qubit_count
        initial_state = np.zeros(dim, dtype=object)
        
        # The amplitude of each basis state in |+⟩^⊗n is 1/sqrt(2^n)
        amplitude = ComplexFP16(1.0 / np.sqrt(2 ** self.qubit_count), 0.0)
        
        for i in range(dim):
            initial_state[i] = amplitude
        
        return initial_state
    
    def get_problem_unitary(self, gamma: float):
        """Create the problem Hamiltonian unitary exp(-i * gamma * H_problem)
        
        Args:
            gamma: The problem Hamiltonian angle parameter
            
        Returns:
            Matrix representing the unitary
        """
        return qaoa_problem_kernel_fp16(gamma, self.edges, self.qubit_count)
    
    def get_mixer_unitary(self, beta: float):
        """Create the mixer Hamiltonian unitary exp(-i * beta * H_mixer)
        
        Args:
            beta: The mixer Hamiltonian angle parameter
            
        Returns:
            Matrix representing the unitary
        """
        return qaoa_mixer_kernel_fp16(beta, self.qubit_count)
    
    def apply_qaoa_circuit(self, parameters: List[float]):
        """Apply the QAOA circuit with given parameters
        
        Args:
            parameters: List of parameters [gamma_1, gamma_2, ..., beta_1, beta_2, ...]
            
        Returns:
            The output state vector
        """
        # Create the initial state
        state = self.create_initial_state()
        
        # Apply each QAOA layer
        for layer in range(self.layer_count):
            gamma = parameters[layer]
            beta = parameters[layer + self.layer_count]
            
            # Get the problem unitary for this layer
            problem_unitary = self.get_problem_unitary(gamma)
            
            # Apply the problem unitary
            intermediate_state = np.zeros(len(state), dtype=object)
            for i in range(len(state)):
                intermediate_state[i] = ComplexFP16(0.0, 0.0)
                for j in range(len(state)):
                    intermediate_state[i] = intermediate_state[i] + problem_unitary[i, j] * state[j]
            
            # Get the mixer unitary for this layer
            mixer_unitary = self.get_mixer_unitary(beta)
            
            # Apply the mixer unitary
            state = np.zeros(len(intermediate_state), dtype=object)
            for i in range(len(intermediate_state)):
                state[i] = ComplexFP16(0.0, 0.0)
                for j in range(len(intermediate_state)):
                    state[i] = state[i] + mixer_unitary[i, j] * intermediate_state[j]
        
        return state
    
    def evaluate_expectation(self, state):
        """Evaluate the expectation value of the Hamiltonian for a given state
        
        Args:
            state: The state to evaluate
            
        Returns:
            Expectation value as a ComplexFP16
        """
        expectation = ComplexFP16(0.0, 0.0)
        dim = 2 ** self.qubit_count
        
        # For each Pauli string in the Hamiltonian
        for pauli_string, coefficient in self.hamiltonian.items():
            # For the identity operator
            if pauli_string == 'I' * self.qubit_count:
                # The expectation value is just the coefficient
                for i in range(dim):
                    expectation += coefficient * state[i] * state[i]
            else:
                # For each basis state
                for i in range(dim):
                    bin_i = format(i, f'0{self.qubit_count}b')
                    
                    # Compute the result of applying this Pauli string to |i⟩
                    target_idx = i
                    phase = ComplexFP16(1.0, 0.0)
                    
                    # Apply each Pauli operator
                    for q in range(self.qubit_count):
                        if pauli_string[q] == 'X':
                            # X flips the bit at position q
                            target_idx ^= (1 << (self.qubit_count - 1 - q))
                        elif pauli_string[q] == 'Z':
                            # Z adds a phase -1 if bit q is 1
                            if bin_i[q] == '1':
                                phase = phase * ComplexFP16(-1.0, 0.0)
                    
                    # Add the contribution to the expectation value
                    expectation += coefficient * state[i] * phase * state[target_idx]
        
        return expectation
    
    def objective_function(self, parameters: List[float]):
        """Compute the objective function (energy) for given parameters
        
        Args:
            parameters: The QAOA parameters
            
        Returns:
            The energy value as a float
        """
        # Apply the QAOA circuit
        state = self.apply_qaoa_circuit(parameters)
        
        # Evaluate the expectation value
        expectation = self.evaluate_expectation(state)
        
        # Return the real part as a float
        return float(expectation.real)
    
    def optimize(self, iterations: int = 100):
        """Optimize the QAOA parameters using a simple grid search
        
        Args:
            iterations: Number of iterations
            
        Returns:
            Tuple of (optimal energy, optimal parameters)
        """
        # Start with random parameters
        best_params = np.random.uniform(-np.pi, np.pi, self.parameter_count)
        best_energy = self.objective_function(best_params)
        
        print(f"Initial parameters: {best_params}")
        print(f"Initial energy: {best_energy}")
        
        # Perform simple optimization
        for i in range(iterations):
            # Try a small random perturbation
            new_params = best_params + np.random.uniform(-0.1, 0.1, self.parameter_count)
            new_energy = self.objective_function(new_params)
            
            # Keep the better parameters
            if new_energy < best_energy:
                best_energy = new_energy
                best_params = new_params
                print(f"Iteration {i+1}: Energy = {best_energy}, Parameters = {best_params}")
        
        return best_energy, best_params
    
    def sample(self, parameters: List[float] = None, shots: int = 1000):
        """Sample from the QAOA circuit
        
        Args:
            parameters: The QAOA parameters (if None, uses optimized parameters)
            shots: Number of shots to sample
            
        Returns:
            Dictionary of bitstrings and their counts
        """
        if parameters is None:
            # Optimize if not already done
            if not hasattr(self, 'optimal_parameters'):
                self.optimal_energy, self.optimal_parameters = self.optimize()
            parameters = self.optimal_parameters
        
        # Apply the QAOA circuit
        state = self.apply_qaoa_circuit(parameters)
        
        # Convert to complex64 for sampling
        state_complex = np.array([x.to_complex64() for x in state])
        
        # Calculate probabilities
        probabilities = np.abs(state_complex) ** 2
        
        # Normalize probabilities
        probabilities /= np.sum(probabilities)
        
        # Sample according to probabilities
        outcomes = np.random.choice(2 ** self.qubit_count, size=shots, p=probabilities)
        
        # Convert to bitstrings and count
        counts = {}
        for outcome in outcomes:
            bitstring = format(outcome, f'0{self.qubit_count}b')
            if bitstring in counts:
                counts[bitstring] += 1
            else:
                counts[bitstring] = 1
        
        # Sort by counts (highest to lowest)
        sorted_counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
        
        return sorted_counts
    
    def get_max_cut(self, counts=None):
        """Get the maximum cut from sampling results
        
        Args:
            counts: Measurement counts (if None, performs sampling)
            
        Returns:
            Bitstring representing the max cut
        """
        if counts is None:
            counts = self.sample()
        
        # The most frequent outcome is our candidate for the max cut
        max_cut = max(counts, key=counts.get)
        
        # Calculate the cut value
        cut_value = 0
        for edge in self.edges:
            u, v = edge
            # If the nodes are in different partitions, add 1 to the cut value
            if max_cut[u] != max_cut[v]:
                cut_value += 1
        
        print(f"Max cut: {max_cut}")
        print(f"Cut value: {cut_value}")
        
        return max_cut, cut_value
    
    def run(self):
        """Run the complete QAOA algorithm: optimize, sample, and return max cut
        
        Returns:
            Tuple of (max cut bitstring, cut value)
        """
        # Optimize parameters
        self.optimal_energy, self.optimal_parameters = self.optimize()
        
        # Sample using optimal parameters
        counts = self.sample(self.optimal_parameters)
        
        # Get and return max cut
        return self.get_max_cut(counts)
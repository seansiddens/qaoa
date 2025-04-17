import numpy as np
import cudaq
from cudaq import spin
from typing import List, Tuple, Dict

class cuQAOA:
    """QAOA implementation using CUDA-Q for solving the Max-Cut problem on a graph"""
    
    def __init__(self, nodes: List[int], edges: List[List[int]], layer_count: int = 2, seed: int = 13):
        self.nodes = nodes
        self.edges = edges
            
        # Extract edges as separate source and target lists for broadcasting into cudaq.kernel
        self.edges_src = [edge[0] for edge in self.edges]
        self.edges_tgt = [edge[1] for edge in self.edges]
        
        # Set problem parameters
        self.qubit_count = len(self.nodes)
        self.layer_count = layer_count  # Default - can be modified
        self.parameter_count = 2 * self.layer_count
        
        # Set random seed for reproducibility
        cudaq.set_random_seed(seed)
        np.random.seed(seed)
        
        # Set default target
        self.set_target('qpp-cpu')
        
        # Initialize optimizer
        self.initialize_optimizer()
        
        # Create Hamiltonian
        self.hamiltonian = self.hamiltonian_max_cut()
        
    def set_target(self, target_name: str):
        """Set the CUDA-Q target
        
        Args:
            target_name: Name of the target ('qpp-cpu' or 'nvidia')
        """
        cudaq.set_target(target_name)
    
    def initialize_optimizer(self):
        """Initialize the optimizer with random initial parameters"""
        self.optimizer = cudaq.optimizers.NelderMead()
        self.optimizer.initial_parameters = np.random.uniform(-np.pi / 8, np.pi / 8,
                                                           self.parameter_count)
        print("Initial parameters = ", self.optimizer.initial_parameters)
    
    @staticmethod
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
    
    def hamiltonian_max_cut(self) -> cudaq.SpinOperator:
        """Create the Hamiltonian for finding the max cut for the graph
        
        Returns:
            Hamiltonian for finding the max cut of the graph
        """
        hamiltonian = 0

        for edge in range(len(self.edges_src)):
            qubitu = self.edges_src[edge]
            qubitv = self.edges_tgt[edge]
            # Add a term to the Hamiltonian for the edge (u,v)
            hamiltonian += 0.5 * (spin.z(qubitu) * spin.z(qubitv) -
                                spin.i(qubitu) * spin.i(qubitv))

        return hamiltonian
    
    def objective(self, parameters: List[float]) -> float:
        """Compute the objective function value for given parameters
        
        Args:
            parameters: List of QAOA parameters (gamma, beta)
            
        Returns:
            Expectation value of the Hamiltonian
        """
        return cudaq.observe(self.kernel_qaoa, self.hamiltonian, self.qubit_count, self.layer_count,
                            self.edges_src, self.edges_tgt, parameters).expectation()
    
    def optimize(self) -> Tuple[float, np.ndarray]:
        """Optimize the QAOA parameters
        
        Returns:
            Tuple of (optimal expectation value, optimal parameters)
        """
        # Create a wrapper function for the optimizer that calls the instance method
        def objective_wrapper(params):
            return self.objective(params)
        
        optimal_expectation, optimal_parameters = self.optimizer.optimize(
            dimensions=self.parameter_count, function=objective_wrapper)
        
        # Print results
        print('optimal_expectation =', optimal_expectation)
        print('Therefore, the max cut value is at least ', -1 * optimal_expectation)
        print('optimal_parameters =', optimal_parameters)
        
        # Store for future use
        self.optimal_expectation = optimal_expectation
        self.optimal_parameters = optimal_parameters
        
        return optimal_expectation, optimal_parameters
    
    def sample(self, parameters: List[float] = None) -> Dict[str, int]:
        """Sample measurement outcomes from the QAOA circuit
        
        Args:
            parameters: QAOA parameters to use (if None, uses optimal parameters from previous optimization)
            
        Returns:
            Dictionary of bitstrings and their counts
        """
        if parameters is None:
            # Optimize if not already done
            if not hasattr(self, 'optimal_parameters'):
                self.optimal_expectation, self.optimal_parameters = self.optimize()
            parameters = self.optimal_parameters
        
        # Sample the circuit
        counts = cudaq.sample(self.kernel_qaoa, self.qubit_count, self.layer_count, 
                            self.edges_src, self.edges_tgt, parameters)
        
        # Print counts sorted by frequency (highest to lowest)
        sorted_counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
        print("Measurement counts (sorted by frequency):")
        for bitstring, count in sorted_counts.items():
            print(f"  |{bitstring}⟩: {count}")
        
        return counts
    
    def get_max_cut(self, counts: Dict[str, int] = None) -> str:
        """Get the maximum cut from sampling results
        
        Args:
            counts: Measurement counts (if None, performs sampling)
            
        Returns:
            Bitstring representing the max cut
        """
        if counts is None:
            counts = self.sample()
        
        # Identify the most likely outcome from the sample
        max_cut = max(counts, key=lambda x: counts[x])
        print('The max cut is given by the partition: ', max_cut)
        
        return max_cut
    
    def run(self) -> str:
        """Run the complete QAOA algorithm: optimize, sample, and return max cut
        
        Returns:
            Bitstring representing the max cut
        """
        # Optimize parameters
        self.optimal_expectation, self.optimal_parameters = self.optimize()
        
        # Sample using optimal parameters
        counts = self.sample(self.optimal_parameters)
        
        # Get and return max cut
        return self.get_max_cut(counts)



if __name__ == "__main__":
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

    # Problem parameters
    # The number of qubits we'll need is the same as the number of vertices in our graph
    qubit_count: int = len(nodes)

    # We can set the layer count to be any positive integer.  Larger values will create deeper circuits
    layer_count: int = 2

    # Each layer of the QAOA kernel contains 2 parameters
    parameter_count: int = 2 * layer_count

    seed = 13

    # Create the CUDA-Q QAOA solver
    print("Initializing CUDA-Q QAOA solver...")
    qaoa = cuQAOA(nodes, edges, layer_count, seed)

    # Print the Hamiltonian
    print(qaoa.hamiltonian)
    
    # Continue with CUDA-Q optimization
    print("\n\nRunning CUDA-Q optimization...")
    optimal_expectation, optimal_parameters = qaoa.optimize()
    
    # Sample using optimal parameters
    print("\nSampling with optimal parameters...")
    counts = qaoa.sample(optimal_parameters)
    
    # Get max cut
    max_cut = qaoa.get_max_cut(counts)
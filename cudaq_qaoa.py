import numpy as np
import cudaq
from cudaq import spin
from typing import List, Tuple, Dict

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


@cudaq.kernel
def qaoaProblem(qubit_0: cudaq.qubit, qubit_1: cudaq.qubit, alpha: float):
    """Build the QAOA gate sequence between two qubits that represent an edge of the graph
    Parameters
    ----------
    qubit_0: cudaq.qubit
        Qubit representing the first vertex of an edge
    qubit_1: cudaq.qubit
        Qubit representing the second vertex of an edge
    thetas: List[float]
        Free variable

    Returns
    -------
    cudaq.Kernel
        Subcircuit of the problem kernel for Max-Cut of the graph with a given edge
    """
    x.ctrl(qubit_0, qubit_1)
    rz(2.0 * alpha, qubit_1)
    x.ctrl(qubit_0, qubit_1)


# We now define the kernel_qaoa function which will be the QAOA circuit for our graph
# Since the QAOA circuit for max cut depends on the structure of the graph,
# we'll feed in global concrete variable values into the kernel_qaoa function for the qubit_count, layer_count, edges_src, edges_tgt.
# The types for these variables are restricted to Quake Values (e.g. qubit, int, List[int], ...)
# The thetas plaeholder will be our free parameters
@cudaq.kernel
def kernel_qaoa(qubit_count: int, layer_count: int, edges_src: List[int],
                edges_tgt: List[int], thetas: List[float]):
    """Build the QAOA circuit for max cut of the graph with given edges and nodes
    Parameters
    ----------
    qubit_count: int
        Number of qubits in the circuit, which is the same as the number of nodes in our graph
    layer_count : int
        Number of layers in the QAOA kernel
    edges_src: List[int]
        List of the first (source) node listed in each edge of the graph, when the edges of the graph are listed as pairs of nodes
    edges_tgt: List[int]
        List of the second (target) node listed in each edge of the graph, when the edges of the graph are listed as pairs of nodes
    thetas: List[float]
        Free variables to be optimized

    Returns
    -------
    cudaq.Kernel
        QAOA circuit for Max-Cut for max cut of the graph with given edges and nodes
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
            qaoaProblem(qreg[qubitu], qreg[qubitv], thetas[i])
        # Add the mixer kernel to each layer
        for j in range(qubit_count):
            rx(2.0 * thetas[i + layer_count], qreg[j])

            
# The problem Hamiltonian
# Define a function to generate the Hamiltonian for a max cut problem using the graph
# with the given edges


def hamiltonian_max_cut(edges_src, edges_tgt):
    """Hamiltonian for finding the max cut for the graph with given edges and nodes

    Parameters
    ----------
    edges_src: List[int]
        List of the first (source) node listed in each edge of the graph, when the edges of the graph are listed as pairs of nodes
    edges_tgt: List[int]
        List of the second (target) node listed in each edge of the graph, when the edges of the graph are listed as pairs of nodes

    Returns
    -------
    cudaq.SpinOperator
        Hamiltonian for finding the max cut of the graph with given edges
    """

    hamiltonian = 0

    for edge in range(len(edges_src)):

        qubitu = edges_src[edge]
        qubitv = edges_tgt[edge]
        # Add a term to the Hamiltonian for the edge (u,v)
        hamiltonian += 0.5 * (spin.z(qubitu) * spin.z(qubitv) -
                              spin.i(qubitu) * spin.i(qubitv))

    return hamiltonian


# Specify the optimizer and its initial parameters.
cudaq.set_random_seed(13)
optimizer = cudaq.optimizers.NelderMead()
np.random.seed(13)
optimizer.initial_parameters = np.random.uniform(-np.pi / 8, np.pi / 8,
                                                 parameter_count)
print("Initial parameters = ", optimizer.initial_parameters)

#cudaq.set_target('nvidia')
cudaq.set_target('qpp-cpu')

# Generate the Hamiltonian for our graph
hamiltonian = hamiltonian_max_cut(edges_src, edges_tgt)
print(hamiltonian)

# Define the objective, return `<state(params) | H | state(params)>`
# Note that in the `observe` call we list the kernel, the hamiltonian, and then the concrete global variable values of our kernel
# followed by the parameters to be optimized.


def objective(parameters):
    return cudaq.observe(kernel_qaoa, hamiltonian, qubit_count, layer_count,
                         edges_src, edges_tgt, parameters).expectation()


# Optimize!
optimal_expectation, optimal_parameters = optimizer.optimize(
    dimensions=parameter_count, function=objective)

# Alternatively we can use the vqe call (just comment out the above code and uncomment the code below)
# optimal_expectation, optimal_parameters = cudaq.vqe(
#    kernel=kernel_qaoa,
#    spin_operator=hamiltonian,
#    argument_mapper=lambda parameter_vector: (qubit_count, layer_count, edges_src, edges_tgt, parameter_vector),
#    optimizer=optimizer,
#    parameter_count=parameter_count)

print('optimal_expectation =', optimal_expectation)
print('Therefore, the max cut value is at least ', -1 * optimal_expectation)
print('optimal_parameters =', optimal_parameters)


# Sample the circuit using the optimized parameters
# Since our kernel has more than one argument, we need to list the values for each of these variables in order in the `sample` call.
counts = cudaq.sample(kernel_qaoa, qubit_count, layer_count, edges_src,
                      edges_tgt, optimal_parameters)
print(counts)

# Identify the most likely outcome from the sample
max_cut = max(counts, key=lambda x: counts[x])
print('The max cut is given by the partition: ',
      max(counts, key=lambda x: counts[x]))


class cuQAOA:
    """QAOA implementation using CUDA-Q for solving the Max-Cut problem on a graph"""
    
    def __init__(self, nodes: List[int] = None, edges: List[List[int]] = None):
        """Initialize the QAOA solver with a graph
        
        Args:
            nodes: List of nodes in the graph (optional)
            edges: List of edges in the graph, where each edge is [source, target] (optional)
        """
        # Set default graph if not provided - this is the example from the original code
        if nodes is None:
            # The graph below illustrates how QAOA can be used to solve a max cut problem
            #       v1  0--------------0 v2
            #           |              | \
            #           |              |  \
            #           |              |   \
            #           |              |    \
            #       v0  0--------------0 v3-- 0 v4
            # The max cut solutions are 01011, 10100, 01010, 10101.
            self.nodes = [0, 1, 2, 3, 4]
            self.edges = [[0, 1], [1, 2], [2, 3], [3, 0], [2, 4], [3, 4]]
        else:
            self.nodes = nodes
            self.edges = edges
            
        # Extract edges as separate source and target lists for broadcasting into cudaq.kernel
        self.edges_src = [edge[0] for edge in self.edges]
        self.edges_tgt = [edge[1] for edge in self.edges]
        
        # Set problem parameters
        self.qubit_count = len(self.nodes)
        self.layer_count = 2  # Default - can be modified
        self.parameter_count = 2 * self.layer_count
        
        # Set random seed for reproducibility
        cudaq.set_random_seed(13)
        np.random.seed(13)
        
        # Set default target
        self.set_target('qpp-cpu')
        
        # Initialize optimizer
        self.initialize_optimizer()
        
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
    
    def qaoaProblem(self, qubit_0: cudaq.qubit, qubit_1: cudaq.qubit, alpha: float):
        """Build the QAOA gate sequence between two qubits that represent an edge of the graph
        
        Args:
            qubit_0: Qubit representing the first vertex of an edge
            qubit_1: Qubit representing the second vertex of an edge
            alpha: Angle parameter
        """
        x.ctrl(qubit_0, qubit_1)
        rz(2.0 * alpha, qubit_1)
        x.ctrl(qubit_0, qubit_1)
    
    @cudaq.kernel
    def kernel_qaoa(self, qubit_count: int, layer_count: int, edges_src: List[int],
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
        hamiltonian = self.hamiltonian_max_cut()
        return cudaq.observe(self.kernel_qaoa, hamiltonian, self.qubit_count, self.layer_count,
                            self.edges_src, self.edges_tgt, parameters).expectation()
    
    def optimize(self) -> Tuple[float, np.ndarray]:
        """Optimize the QAOA parameters
        
        Returns:
            Tuple of (optimal expectation value, optimal parameters)
        """
        optimal_expectation, optimal_parameters = self.optimizer.optimize(
            dimensions=self.parameter_count, function=self.objective)
        
        # Print results
        print('optimal_expectation =', optimal_expectation)
        print('Therefore, the max cut value is at least ', -1 * optimal_expectation)
        print('optimal_parameters =', optimal_parameters)
        
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
        print(counts)
        
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


# Example usage (if run directly)
if __name__ == "__main__":
    # Create and run the QAOA solver
    qaoa = cuQAOA()
    
    # Print the Hamiltonian
    print(qaoa.hamiltonian_max_cut())
    
    # Run the complete algorithm
    max_cut = qaoa.run()
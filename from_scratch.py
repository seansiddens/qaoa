import numpy as np
import cudaq
from typing import List

# ---------- NumPy Implementation ----------

def rx_gate_numpy(theta, dtype=np.complex64):
    """Returns the RX gate matrix for a given theta using NumPy"""
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)
    return np.array([
        [cos, -1j * sin],
        [-1j * sin, cos]
    ], dtype=dtype)

def simulate_numpy_rx(theta, dtype=np.complex64):
    """Apply RX to |0⟩ and return resulting state vector"""
    input_state = np.array([1.0, 0.0], dtype=dtype)
    rx = rx_gate_numpy(theta, dtype=dtype)
    result = rx @ input_state
    return result

def h_gate_numpy(dtype=np.complex64):
    """Returns the Hadamard gate matrix using NumPy"""
    return np.array([
        [1.0, 1.0],
        [1.0, -1.0]
    ], dtype=dtype) / np.sqrt(2)

def simulate_numpy_h(dtype=np.complex64):
    """Apply H to |0⟩ and return resulting state vector"""
    input_state = np.array([1.0, 0.0], dtype=dtype)
    h = h_gate_numpy(dtype=dtype)
    result = h @ input_state
    return result

def rz_gate_numpy(theta, dtype=np.complex64):
    """Returns the RZ gate matrix for a given theta using NumPy"""
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=dtype)

def simulate_numpy_rz(theta, dtype=np.complex64):
    """Apply RZ to |0⟩ and return resulting state vector"""
    input_state = np.array([1.0, 0.0], dtype=dtype)
    rz = rz_gate_numpy(theta, dtype=dtype)
    result = rz @ input_state
    return result

def random_state_numpy(dtype=np.complex64):
    """Generate a random normalized quantum state vector"""
    # Generate random complex numbers
    real_part = np.random.randn(2)
    imag_part = np.random.randn(2)
    state = real_part + 1j * imag_part
    # Normalize
    norm = np.sqrt(np.sum(np.abs(state)**2))
    return (state / norm).astype(dtype)

# ---------- FP16 Complex Implementation ----------

class ComplexFP16:
    """A complex number represented by two fp16 values (real and imaginary parts)"""
    def __init__(self, real, imag=0.0):
        self.real = np.float16(real)
        self.imag = np.float16(imag)
    
    def __add__(self, other):
        if isinstance(other, ComplexFP16):
            return ComplexFP16(self.real + other.real, self.imag + other.imag)
        elif isinstance(other, (int, float)):
            return ComplexFP16(self.real + other, self.imag)
        else:
            return NotImplemented
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, ComplexFP16):
            return ComplexFP16(self.real - other.real, self.imag - other.imag)
        elif isinstance(other, (int, float)):
            return ComplexFP16(self.real - other, self.imag)
        else:
            return NotImplemented
    
    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return ComplexFP16(other - self.real, -self.imag)
        else:
            return NotImplemented
    
    def __mul__(self, other):
        if isinstance(other, ComplexFP16):
            # (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            real = self.real * other.real - self.imag * other.imag
            imag = self.real * other.imag + self.imag * other.real
            return ComplexFP16(real, imag)
        elif isinstance(other, (int, float)):
            return ComplexFP16(self.real * other, self.imag * other)
        else:
            return NotImplemented
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return ComplexFP16(self.real / other, self.imag / other)
        else:
            return NotImplemented
    
    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            # a/(b+ci) = a(b-ci)/(b^2+c^2)
            denominator = self.real * self.real + self.imag * self.imag
            real = other * self.real / denominator
            imag = -other * self.imag / denominator
            return ComplexFP16(real, imag)
        else:
            return NotImplemented
    
    def __repr__(self):
        return f"ComplexFP16({self.real}, {self.imag})"
    
    def to_complex64(self):
        """Convert to numpy complex64 for compatibility with other functions"""
        return np.complex64(self.real.astype(np.float32) + 1j * self.imag.astype(np.float32))

def rx_gate_fp16(theta):
    """Returns the RX gate matrix using FP16 complex numbers"""
    cos = np.float16(np.cos(theta / 2))
    sin = np.float16(np.sin(theta / 2))
    
    return np.array([
        [ComplexFP16(cos), ComplexFP16(0, -sin)],
        [ComplexFP16(0, -sin), ComplexFP16(cos)]
    ])

def simulate_fp16_rx(theta):
    """Apply RX to |0⟩ using FP16 complex numbers and return resulting state vector"""
    input_state = np.array([ComplexFP16(1.0), ComplexFP16(0.0)])
    rx = rx_gate_fp16(theta)
    
    # Manual matrix multiplication with FP16 complex numbers
    result = np.array([
        rx[0, 0] * input_state[0] + rx[0, 1] * input_state[1],
        rx[1, 0] * input_state[0] + rx[1, 1] * input_state[1]
    ])
    
    return result

def h_gate_fp16():
    """Returns the Hadamard gate matrix using FP16 complex numbers"""
    # 1/sqrt(2) is approximately 0.7071067811865476
    h_factor = np.float16(0.7071067811865476)
    
    return np.array([
        [ComplexFP16(h_factor), ComplexFP16(h_factor)],
        [ComplexFP16(h_factor), ComplexFP16(-h_factor)]
    ])

def simulate_fp16_h():
    """Apply H to |0⟩ using FP16 complex numbers and return resulting state vector"""
    input_state = np.array([ComplexFP16(1.0), ComplexFP16(0.0)])
    h = h_gate_fp16()
    
    # Manual matrix multiplication with FP16 complex numbers
    result = np.array([
        h[0, 0] * input_state[0] + h[0, 1] * input_state[1],
        h[1, 0] * input_state[0] + h[1, 1] * input_state[1]
    ])
    
    return result

def rz_gate_fp16(theta):
    """Returns the RZ gate matrix using FP16 complex numbers"""
    # Calculate exp(-i*theta/2) and exp(i*theta/2)
    cos_neg = np.float16(np.cos(-theta / 2))
    sin_neg = np.float16(np.sin(-theta / 2))
    cos_pos = np.float16(np.cos(theta / 2))
    sin_pos = np.float16(np.sin(theta / 2))
    
    return np.array([
        [ComplexFP16(cos_neg, sin_neg), ComplexFP16(0.0)],
        [ComplexFP16(0.0), ComplexFP16(cos_pos, sin_pos)]
    ])

def simulate_fp16_rz(theta):
    """Apply RZ to |0⟩ using FP16 complex numbers and return resulting state vector"""
    input_state = np.array([ComplexFP16(1.0), ComplexFP16(0.0)])
    rz = rz_gate_fp16(theta)
    
    # Manual matrix multiplication with FP16 complex numbers
    result = np.array([
        rz[0, 0] * input_state[0] + rz[0, 1] * input_state[1],
        rz[1, 0] * input_state[0] + rz[1, 1] * input_state[1]
    ])
    
    return result

def random_state_fp16():
    """Generate a random normalized quantum state vector using FP16 complex numbers"""
    # Generate random real and imaginary parts
    real_part = np.random.randn(2).astype(np.float16)
    imag_part = np.random.randn(2).astype(np.float16)
    
    # Create complex numbers
    state = np.array([
        ComplexFP16(real_part[0], imag_part[0]),
        ComplexFP16(real_part[1], imag_part[1])
    ])
    
    # Normalize
    norm_squared = state[0].real * state[0].real + state[0].imag * state[0].imag + \
                   state[1].real * state[1].real + state[1].imag * state[1].imag
    norm = np.float16(np.sqrt(float(norm_squared)))
    
    return np.array([
        ComplexFP16(state[0].real / norm, state[0].imag / norm),
        ComplexFP16(state[1].real / norm, state[1].imag / norm)
    ])

# ---------- CUDA-Q Implementation ----------

def simulate_cudaq_rx(theta):
    """Use CUDA-Q to apply RX(θ) to |0⟩"""
    @cudaq.kernel
    def rx_kernel(theta: float):
        q = cudaq.qubit()
        rx(theta, q)

    # Retrieve state vector
    state = cudaq.get_state(rx_kernel, theta)
    return np.array(state)

def simulate_cudaq_h():
    """Use CUDA-Q to apply H to |0⟩"""
    @cudaq.kernel
    def h_kernel():
        q = cudaq.qubit()
        h(q)

    # Retrieve state vector
    state = cudaq.get_state(h_kernel)
    return np.array(state)

def simulate_cudaq_rz(theta):
    """Use CUDA-Q to apply RZ(θ) to |0⟩"""
    @cudaq.kernel
    def rz_kernel(theta: float):
        q = cudaq.qubit()
        rz(theta, q)

    # Retrieve state vector
    state = cudaq.get_state(rz_kernel, theta)
    return np.array(state)

# ---------- Fidelity Comparison ----------

def fidelity(a, b):
    """Compute the fidelity (|⟨a|b⟩|^2) between two state vectors"""
    return np.abs(np.vdot(a, b))**2

# ---------- Multi-Qubit Gate Operations ----------

def cz_gate_numpy(dtype=np.complex64):
    """Returns the controlled-Z gate matrix using NumPy"""
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, -1.0]
    ], dtype=dtype)

def cz_gate_fp16():
    """Returns the controlled-Z gate matrix using FP16 complex numbers"""
    return np.array([
        [ComplexFP16(1.0), ComplexFP16(0.0), ComplexFP16(0.0), ComplexFP16(0.0)],
        [ComplexFP16(0.0), ComplexFP16(1.0), ComplexFP16(0.0), ComplexFP16(0.0)],
        [ComplexFP16(0.0), ComplexFP16(0.0), ComplexFP16(1.0), ComplexFP16(0.0)],
        [ComplexFP16(0.0), ComplexFP16(0.0), ComplexFP16(0.0), ComplexFP16(-1.0)]
    ])

def simulate_numpy_cz(input_state=None, dtype=np.complex64):
    """Apply CZ to a 2-qubit input state (defaults to |00⟩) and return resulting state vector"""
    if input_state is None:
        # Default to |00⟩ state
        input_state = np.zeros(4, dtype=dtype)
        input_state[0] = 1.0
    cz = cz_gate_numpy(dtype=dtype)
    result = cz @ input_state
    return result

def simulate_fp16_cz(input_state=None):
    """Apply CZ to a 2-qubit input state (defaults to |00⟩) using FP16 complex numbers and return resulting state vector"""
    if input_state is None:
        # Default to |00⟩ state
        input_state = np.array([
            ComplexFP16(1.0), 
            ComplexFP16(0.0), 
            ComplexFP16(0.0), 
            ComplexFP16(0.0)
        ])
    cz = cz_gate_fp16()
    
    # Manual matrix multiplication with FP16 complex numbers
    result = np.array([
        cz[0, 0] * input_state[0] + cz[0, 1] * input_state[1] + cz[0, 2] * input_state[2] + cz[0, 3] * input_state[3],
        cz[1, 0] * input_state[0] + cz[1, 1] * input_state[1] + cz[1, 2] * input_state[2] + cz[1, 3] * input_state[3],
        cz[2, 0] * input_state[0] + cz[2, 1] * input_state[1] + cz[2, 2] * input_state[2] + cz[2, 3] * input_state[3],
        cz[3, 0] * input_state[0] + cz[3, 1] * input_state[1] + cz[3, 2] * input_state[2] + cz[3, 3] * input_state[3]
    ])
    
    return result

def simulate_cudaq_cz():
    """Use CUDA-Q to apply CZ to |00⟩"""
    @cudaq.kernel
    def cz_kernel():
        q0 = cudaq.qubit()
        q1 = cudaq.qubit()
        z.ctrl(q0, q1)  # Controlled-Z gate

    # Retrieve state vector
    state = cudaq.get_state(cz_kernel)
    return np.array(state)

def test_cz_gate():
    """Test the CZ gate implementation"""
    print("\nTesting CZ gate implementation...")
    
    # Apply CZ to |00⟩
    result_np = simulate_numpy_cz(dtype=np.complex64)
    result_fp16 = simulate_fp16_cz()
    result_cudaq = simulate_cudaq_cz()
    
    # Convert FP16 results to complex64 for comparison
    result_fp16_complex = np.array([x.to_complex64() for x in result_fp16])
    
    # Calculate fidelity
    fid_np_cudaq = fidelity(result_np, result_cudaq)
    fid_fp16_np = fidelity(result_fp16_complex, result_np)
    fid_fp16_cudaq = fidelity(result_fp16_complex, result_cudaq)
    
    print("CZ gate results (|00⟩ input):")
    print("CUDA-Q Result:", result_cudaq)
    print("NumPy Result:", result_np)
    print("FP16 Result:", [str(x) for x in result_fp16])
    print(f"Fidelity between NumPy and CUDA-Q: {fid_np_cudaq:.6f}")
    print(f"Fidelity between FP16 and CUDA-Q: {fid_fp16_cudaq:.6f}")
    print(f"Fidelity between FP16 and NumPy: {fid_fp16_np:.6f}")
    
    # Now test with the |11⟩ state
    input_state_np = np.zeros(4, dtype=np.complex64)
    input_state_np[3] = 1.0  # |11⟩ state
    
    input_state_fp16 = np.array([
        ComplexFP16(0.0), 
        ComplexFP16(0.0), 
        ComplexFP16(0.0), 
        ComplexFP16(1.0)
    ])
    
    result_np_11 = simulate_numpy_cz(input_state_np, dtype=np.complex64)
    result_fp16_11 = simulate_fp16_cz(input_state_fp16)
    
    # Convert FP16 results to complex64 for comparison
    result_fp16_complex_11 = np.array([x.to_complex64() for x in result_fp16_11])
    
    # Calculate fidelity
    fid_fp16_np_11 = fidelity(result_fp16_complex_11, result_np_11)
    
    print("\nCZ gate results (|11⟩ input):")
    print("NumPy Result:", result_np_11)
    print("FP16 Result:", [str(x) for x in result_fp16_11])
    print(f"Fidelity between FP16 and NumPy: {fid_fp16_np_11:.6f}")

# ---------- Tensor Product Operations ----------

def tensor_product_numpy(matrix_a, matrix_b, dtype=np.complex64):
    """Compute tensor product of two matrices using NumPy"""
    return np.kron(matrix_a, matrix_b).astype(dtype)

def tensor_product_fp16(matrix_a, matrix_b):
    """Compute tensor product of two matrices using FP16 complex numbers"""
    rows_a, cols_a = matrix_a.shape
    rows_b, cols_b = matrix_b.shape
    
    result = np.empty((rows_a * rows_b, cols_a * cols_b), dtype=object)
    
    for i in range(rows_a):
        for j in range(cols_a):
            for k in range(rows_b):
                for l in range(cols_b):
                    result[i * rows_b + k, j * cols_b + l] = matrix_a[i, j] * matrix_b[k, l]
    
    return result

def test_tensor_product():
    """Test the tensor product implementation"""
    print("\nTesting tensor product implementation...")
    
    # Test H ⊗ I
    h_gate = h_gate_numpy()
    i_gate = np.eye(2, dtype=np.complex64)
    
    h_i_numpy = tensor_product_numpy(h_gate, i_gate)
    
    h_gate_fp16_matrix = h_gate_fp16()
    i_gate_fp16 = np.array([
        [ComplexFP16(1.0), ComplexFP16(0.0)],
        [ComplexFP16(0.0), ComplexFP16(1.0)]
    ])
    
    h_i_fp16 = tensor_product_fp16(h_gate_fp16_matrix, i_gate_fp16)
    
    # Convert FP16 results to complex64 for comparison
    h_i_fp16_complex = np.zeros(h_i_fp16.shape, dtype=np.complex64)
    for i in range(h_i_fp16.shape[0]):
        for j in range(h_i_fp16.shape[1]):
            h_i_fp16_complex[i, j] = h_i_fp16[i, j].to_complex64()
    
    # Calculate norm difference
    norm_diff = np.linalg.norm(h_i_numpy - h_i_fp16_complex)
    
    print("H ⊗ I tensor product:")
    print("NumPy result shape:", h_i_numpy.shape)
    print("FP16 result shape:", h_i_fp16.shape)
    print(f"Norm difference: {norm_diff}")
    
    # Test applying H ⊗ I to |00⟩
    state_numpy = np.zeros(4, dtype=np.complex64)
    state_numpy[0] = 1.0  # |00⟩
    
    state_fp16 = np.array([
        ComplexFP16(1.0),
        ComplexFP16(0.0),
        ComplexFP16(0.0),
        ComplexFP16(0.0)
    ])
    
    # Apply the tensor product matrices
    result_numpy = h_i_numpy @ state_numpy
    
    # Manual matrix multiplication for FP16
    result_fp16 = np.zeros(4, dtype=object)
    for i in range(4):
        for j in range(4):
            result_fp16[i] = result_fp16[i] + h_i_fp16[i, j] * state_fp16[j]
    
    # Convert FP16 result to complex64 for comparison
    result_fp16_complex = np.array([x.to_complex64() for x in result_fp16])
    
    # Calculate fidelity
    fid = fidelity(result_numpy, result_fp16_complex)
    
    print("\nApplying H ⊗ I to |00⟩:")
    print("NumPy result:", result_numpy)
    print("FP16 result:", [str(x) for x in result_fp16])
    print(f"Fidelity: {fid:.6f}")

# ---------- CNOT and Multi-Qubit Operations ----------

def cnot_gate_numpy(dtype=np.complex64):
    """Returns the CNOT gate matrix using NumPy"""
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
    ], dtype=dtype)

def cnot_gate_fp16():
    """Returns the CNOT gate matrix using FP16 complex numbers"""
    return np.array([
        [ComplexFP16(1.0), ComplexFP16(0.0), ComplexFP16(0.0), ComplexFP16(0.0)],
        [ComplexFP16(0.0), ComplexFP16(1.0), ComplexFP16(0.0), ComplexFP16(0.0)],
        [ComplexFP16(0.0), ComplexFP16(0.0), ComplexFP16(0.0), ComplexFP16(1.0)],
        [ComplexFP16(0.0), ComplexFP16(0.0), ComplexFP16(1.0), ComplexFP16(0.0)]
    ])

def simulate_numpy_cnot(input_state=None, dtype=np.complex64):
    """Apply CNOT to a 2-qubit input state (defaults to |00⟩) and return resulting state vector"""
    if input_state is None:
        # Default to |00⟩ state
        input_state = np.zeros(4, dtype=dtype)
        input_state[0] = 1.0
    cnot = cnot_gate_numpy(dtype=dtype)
    result = cnot @ input_state
    return result

def simulate_fp16_cnot(input_state=None):
    """Apply CNOT to a 2-qubit input state (defaults to |00⟩) using FP16 complex numbers and return resulting state vector"""
    if input_state is None:
        # Default to |00⟩ state
        input_state = np.array([
            ComplexFP16(1.0), 
            ComplexFP16(0.0), 
            ComplexFP16(0.0), 
            ComplexFP16(0.0)
        ])
    cnot = cnot_gate_fp16()
    
    # Manual matrix multiplication with FP16 complex numbers
    result = np.array([
        cnot[0, 0] * input_state[0] + cnot[0, 1] * input_state[1] + cnot[0, 2] * input_state[2] + cnot[0, 3] * input_state[3],
        cnot[1, 0] * input_state[0] + cnot[1, 1] * input_state[1] + cnot[1, 2] * input_state[2] + cnot[1, 3] * input_state[3],
        cnot[2, 0] * input_state[0] + cnot[2, 1] * input_state[1] + cnot[2, 2] * input_state[2] + cnot[2, 3] * input_state[3],
        cnot[3, 0] * input_state[0] + cnot[3, 1] * input_state[1] + cnot[3, 2] * input_state[2] + cnot[3, 3] * input_state[3]
    ])
    
    return result

def simulate_cudaq_cnot():
    """Use CUDA-Q to apply CNOT to |00⟩"""
    @cudaq.kernel
    def cnot_kernel():
        q0 = cudaq.qubit()
        q1 = cudaq.qubit()
        x.ctrl(q0, q1)  # CNOT gate

    # Retrieve state vector
    state = cudaq.get_state(cnot_kernel)
    return np.array(state)

def test_cnot_gate():
    """Test the CNOT gate implementation"""
    print("\nTesting CNOT gate implementation...")
    
    # Apply CNOT to |00⟩
    result_np = simulate_numpy_cnot(dtype=np.complex64)
    result_fp16 = simulate_fp16_cnot()
    result_cudaq = simulate_cudaq_cnot()
    
    # Convert FP16 results to complex64 for comparison
    result_fp16_complex = np.array([x.to_complex64() for x in result_fp16])
    
    # Calculate fidelity
    fid_np_cudaq = fidelity(result_np, result_cudaq)
    fid_fp16_np = fidelity(result_fp16_complex, result_np)
    fid_fp16_cudaq = fidelity(result_fp16_complex, result_cudaq)
    
    print("CNOT gate results (|00⟩ input):")
    print("CUDA-Q Result:", result_cudaq)
    print("NumPy Result:", result_np)
    print("FP16 Result:", [str(x) for x in result_fp16])
    print(f"Fidelity between NumPy and CUDA-Q: {fid_np_cudaq:.6f}")
    print(f"Fidelity between FP16 and CUDA-Q: {fid_fp16_cudaq:.6f}")
    print(f"Fidelity between FP16 and NumPy: {fid_fp16_np:.6f}")
    
    # Now test with the |10⟩ state (control = 1, target = 0)
    input_state_np = np.zeros(4, dtype=np.complex64)
    input_state_np[2] = 1.0  # |10⟩ state
    
    input_state_fp16 = np.array([
        ComplexFP16(0.0), 
        ComplexFP16(0.0), 
        ComplexFP16(1.0), 
        ComplexFP16(0.0)
    ])
    
    # Create |10⟩ state in CUDA-Q
    @cudaq.kernel
    def cnot_10_kernel():
        q0 = cudaq.qubit()
        q1 = cudaq.qubit()
        x(q0)  # Flip first qubit to |1⟩
        x.ctrl(q0, q1)  # Apply CNOT
    
    # Apply CNOT to |10⟩
    result_np_10 = simulate_numpy_cnot(input_state_np, dtype=np.complex64)
    result_fp16_10 = simulate_fp16_cnot(input_state_fp16)
    result_cudaq_10 = cudaq.get_state(cnot_10_kernel)
    
    # Convert FP16 results to complex64 for comparison
    result_fp16_complex_10 = np.array([x.to_complex64() for x in result_fp16_10])
    
    # Calculate fidelity
    fid_np_cudaq_10 = fidelity(result_np_10, result_cudaq_10)
    fid_fp16_np_10 = fidelity(result_fp16_complex_10, result_np_10)
    fid_fp16_cudaq_10 = fidelity(result_fp16_complex_10, result_cudaq_10)
    
    print("\nCNOT gate results (|10⟩ input):")
    print("CUDA-Q Result:", result_cudaq_10)
    print("NumPy Result:", result_np_10)
    print("FP16 Result:", [str(x) for x in result_fp16_10])
    print(f"Fidelity between NumPy and CUDA-Q: {fid_np_cudaq_10:.6f}")
    print(f"Fidelity between FP16 and CUDA-Q: {fid_fp16_cudaq_10:.6f}")
    print(f"Fidelity between FP16 and NumPy: {fid_fp16_np_10:.6f}")

# ---------- ZZ Interaction Implementation ----------

def zz_interaction_numpy(gamma, dtype=np.complex64):
    """Returns exp(-i*gamma*Z⊗Z) gate matrix using NumPy"""
    # The ZZ interaction is exp(-i*gamma*Z⊗Z)
    # Z⊗Z has eigenvalues +1 for |00⟩ and |11⟩, and -1 for |01⟩ and |10⟩
    # So exp(-i*gamma*Z⊗Z) adds phase e^(-i*gamma) to |00⟩ and |11⟩, and e^(i*gamma) to |01⟩ and |10⟩
    
    # Calculate exponentials
    exp_neg = np.exp(-1j * gamma)
    exp_pos = np.exp(1j * gamma)
    
    return np.array([
        [exp_neg, 0.0, 0.0, 0.0],
        [0.0, exp_pos, 0.0, 0.0],
        [0.0, 0.0, exp_pos, 0.0],
        [0.0, 0.0, 0.0, exp_neg]
    ], dtype=dtype)

def zz_interaction_fp16(gamma):
    """Returns exp(-i*gamma*Z⊗Z) gate matrix using FP16 complex numbers"""
    # Calculate exponentials
    cos_neg = np.float16(np.cos(-gamma))
    sin_neg = np.float16(np.sin(-gamma))
    cos_pos = np.float16(np.cos(gamma))
    sin_pos = np.float16(np.sin(gamma))
    
    return np.array([
        [ComplexFP16(cos_neg, sin_neg), ComplexFP16(0.0), ComplexFP16(0.0), ComplexFP16(0.0)],
        [ComplexFP16(0.0), ComplexFP16(cos_pos, sin_pos), ComplexFP16(0.0), ComplexFP16(0.0)],
        [ComplexFP16(0.0), ComplexFP16(0.0), ComplexFP16(cos_pos, sin_pos), ComplexFP16(0.0)],
        [ComplexFP16(0.0), ComplexFP16(0.0), ComplexFP16(0.0), ComplexFP16(cos_neg, sin_neg)]
    ])

def simulate_numpy_zz(gamma, input_state=None, dtype=np.complex64):
    """Apply exp(-i*gamma*Z⊗Z) to a 2-qubit input state (defaults to |00⟩) and return resulting state vector"""
    if input_state is None:
        # Default to |00⟩ state
        input_state = np.zeros(4, dtype=dtype)
        input_state[0] = 1.0
    zz = zz_interaction_numpy(gamma, dtype=dtype)
    result = zz @ input_state
    return result

def simulate_fp16_zz(gamma, input_state=None):
    """Apply exp(-i*gamma*Z⊗Z) to a 2-qubit input state (defaults to |00⟩) using FP16 complex numbers and return resulting state vector"""
    if input_state is None:
        # Default to |00⟩ state
        input_state = np.array([
            ComplexFP16(1.0), 
            ComplexFP16(0.0), 
            ComplexFP16(0.0), 
            ComplexFP16(0.0)
        ])
    zz = zz_interaction_fp16(gamma)
    
    # Manual matrix multiplication with FP16 complex numbers
    result = np.array([
        zz[0, 0] * input_state[0] + zz[0, 1] * input_state[1] + zz[0, 2] * input_state[2] + zz[0, 3] * input_state[3],
        zz[1, 0] * input_state[0] + zz[1, 1] * input_state[1] + zz[1, 2] * input_state[2] + zz[1, 3] * input_state[3],
        zz[2, 0] * input_state[0] + zz[2, 1] * input_state[1] + zz[2, 2] * input_state[2] + zz[2, 3] * input_state[3],
        zz[3, 0] * input_state[0] + zz[3, 1] * input_state[1] + zz[3, 2] * input_state[2] + zz[3, 3] * input_state[3]
    ])
    
    return result

def simulate_cudaq_zz(gamma):
    """Use CUDA-Q to apply exp(-i*gamma*Z⊗Z) to |00⟩"""
    @cudaq.kernel
    def zz_kernel(gamma: float):
        q0 = cudaq.qubit()
        q1 = cudaq.qubit()
        
        # exp(-i*gamma*Z⊗Z) can be implemented as: CNOT -> Rz(2*gamma) -> CNOT
        x.ctrl(q0, q1)  # CNOT
        rz(2.0 * gamma, q1)  # Rz(2*gamma)
        x.ctrl(q0, q1)  # CNOT

    # Retrieve state vector
    state = cudaq.get_state(zz_kernel, gamma)
    return np.array(state)

def test_zz_interaction():
    """Test the ZZ interaction implementation"""
    print("\nTesting ZZ interaction implementation...")
    
    # Choose a test angle
    gamma = np.pi / 4
    
    # Apply ZZ to |00⟩
    result_np = simulate_numpy_zz(gamma, dtype=np.complex64)
    result_fp16 = simulate_fp16_zz(gamma)
    result_cudaq = simulate_cudaq_zz(gamma)
    
    # Convert FP16 results to complex64 for comparison
    result_fp16_complex = np.array([x.to_complex64() for x in result_fp16])
    
    # Calculate fidelity
    fid_np_cudaq = fidelity(result_np, result_cudaq)
    fid_fp16_np = fidelity(result_fp16_complex, result_np)
    fid_fp16_cudaq = fidelity(result_fp16_complex, result_cudaq)
    
    print(f"ZZ interaction exp(-i*{gamma}*Z⊗Z) results (|00⟩ input):")
    print("CUDA-Q Result:", result_cudaq)
    print("NumPy Result:", result_np)
    print("FP16 Result:", [str(x) for x in result_fp16])
    print(f"Fidelity between NumPy and CUDA-Q: {fid_np_cudaq:.6f}")
    print(f"Fidelity between FP16 and CUDA-Q: {fid_fp16_cudaq:.6f}")
    print(f"Fidelity between FP16 and NumPy: {fid_fp16_np:.6f}")
    
    # Create equal superposition state |++⟩ = (|00⟩ + |01⟩ + |10⟩ + |11⟩)/2
    @cudaq.kernel
    def zz_plusplus_kernel(gamma: float):
        q0 = cudaq.qubit()
        q1 = cudaq.qubit()
        
        # Create |++⟩ state
        h(q0)
        h(q1)
        
        # Apply ZZ interaction
        x.ctrl(q0, q1)  # CNOT
        rz(2.0 * gamma, q1)  # Rz(2*gamma)
        x.ctrl(q0, q1)  # CNOT

    # Run CUDA-Q with |++⟩ input
    result_cudaq_plusplus = cudaq.get_state(zz_plusplus_kernel, gamma)
    
    # Create |++⟩ state in NumPy
    input_state_np_plusplus = np.ones(4, dtype=np.complex64) / 2.0
    
    # Create |++⟩ state in FP16
    input_state_fp16_plusplus = np.array([
        ComplexFP16(0.5), 
        ComplexFP16(0.5), 
        ComplexFP16(0.5), 
        ComplexFP16(0.5)
    ])
    
    # Apply ZZ interaction to |++⟩
    result_np_plusplus = simulate_numpy_zz(gamma, input_state_np_plusplus, dtype=np.complex64)
    result_fp16_plusplus = simulate_fp16_zz(gamma, input_state_fp16_plusplus)
    
    # Convert FP16 results to complex64 for comparison
    result_fp16_complex_plusplus = np.array([x.to_complex64() for x in result_fp16_plusplus])
    
    # Calculate fidelity
    fid_np_cudaq_plusplus = fidelity(result_np_plusplus, result_cudaq_plusplus)
    fid_fp16_np_plusplus = fidelity(result_fp16_complex_plusplus, result_np_plusplus)
    fid_fp16_cudaq_plusplus = fidelity(result_fp16_complex_plusplus, result_cudaq_plusplus)
    
    print(f"\nZZ interaction exp(-i*{gamma}*Z⊗Z) results (|++⟩ input):")
    print("CUDA-Q Result:", result_cudaq_plusplus)
    print("NumPy Result:", result_np_plusplus)
    print("FP16 Result:", [str(x) for x in result_fp16_plusplus])
    print(f"Fidelity between NumPy and CUDA-Q: {fid_np_cudaq_plusplus:.6f}")
    print(f"Fidelity between FP16 and CUDA-Q: {fid_fp16_cudaq_plusplus:.6f}")
    print(f"Fidelity between FP16 and NumPy: {fid_fp16_np_plusplus:.6f}")

# ---------- Simple QAOA Implementation ----------

def simple_qaoa_circuit_numpy(gamma, beta, dtype=np.complex64):
    """Implement a single layer of QAOA for a 2-qubit system (single edge) using NumPy
    
    Args:
        gamma: Problem Hamiltonian angle
        beta: Mixer Hamiltonian angle
        dtype: NumPy data type
        
    Returns:
        Final state vector after QAOA circuit
    """
    # Start with |00⟩ state
    state = np.zeros(4, dtype=dtype)
    state[0] = 1.0
    
    # Step 1: Apply Hadamard to both qubits (create superposition)
    h_gate = h_gate_numpy(dtype=dtype)
    h_h = tensor_product_numpy(h_gate, h_gate, dtype=dtype)
    state = h_h @ state
    
    # Step 2: Apply problem Hamiltonian exp(-i*gamma*Z⊗Z)
    # For a single edge between qubit 0 and 1, we apply ZZ interaction
    zz = zz_interaction_numpy(gamma, dtype=dtype)
    state = zz @ state
    
    # Step 3: Apply mixer Hamiltonian exp(-i*beta*X⊗I) ⊗ exp(-i*beta*I⊗X)
    # Apply RX(2*beta) to both qubits
    rx_gate = rx_gate_numpy(2.0 * beta, dtype=dtype)
    identity = np.eye(2, dtype=dtype)
    
    # RX on first qubit
    rx_i = tensor_product_numpy(rx_gate, identity, dtype=dtype)
    state = rx_i @ state
    
    # RX on second qubit
    i_rx = tensor_product_numpy(identity, rx_gate, dtype=dtype)
    state = i_rx @ state
    
    return state

def simple_qaoa_circuit_fp16(gamma, beta):
    """Implement a single layer of QAOA for a 2-qubit system (single edge) using FP16
    
    Args:
        gamma: Problem Hamiltonian angle
        beta: Mixer Hamiltonian angle
        
    Returns:
        Final state vector after QAOA circuit
    """
    # Start with |00⟩ state
    state = np.array([
        ComplexFP16(1.0),
        ComplexFP16(0.0),
        ComplexFP16(0.0),
        ComplexFP16(0.0)
    ])
    
    # Step 1: Apply Hadamard to both qubits (create superposition)
    h_gate = h_gate_fp16()
    identity = np.array([
        [ComplexFP16(1.0), ComplexFP16(0.0)],
        [ComplexFP16(0.0), ComplexFP16(1.0)]
    ])
    
    # H on first qubit
    h_i = tensor_product_fp16(h_gate, identity)
    
    # Manual matrix multiplication
    temp_state = np.zeros(4, dtype=object)
    for i in range(4):
        for j in range(4):
            temp_state[i] = temp_state[i] + h_i[i, j] * state[j]
    state = temp_state
    
    # H on second qubit
    i_h = tensor_product_fp16(identity, h_gate)
    
    # Manual matrix multiplication
    temp_state = np.zeros(4, dtype=object)
    for i in range(4):
        for j in range(4):
            temp_state[i] = temp_state[i] + i_h[i, j] * state[j]
    state = temp_state
    
    # Step 2: Apply problem Hamiltonian exp(-i*gamma*Z⊗Z)
    zz = zz_interaction_fp16(gamma)
    
    # Manual matrix multiplication
    temp_state = np.zeros(4, dtype=object)
    for i in range(4):
        for j in range(4):
            temp_state[i] = temp_state[i] + zz[i, j] * state[j]
    state = temp_state
    
    # Step 3: Apply mixer Hamiltonian exp(-i*beta*X⊗I) ⊗ exp(-i*beta*I⊗X)
    rx_gate = rx_gate_fp16(2.0 * beta)
    
    # RX on first qubit
    rx_i = tensor_product_fp16(rx_gate, identity)
    
    # Manual matrix multiplication
    temp_state = np.zeros(4, dtype=object)
    for i in range(4):
        for j in range(4):
            temp_state[i] = temp_state[i] + rx_i[i, j] * state[j]
    state = temp_state
    
    # RX on second qubit
    i_rx = tensor_product_fp16(identity, rx_gate)
    
    # Manual matrix multiplication
    temp_state = np.zeros(4, dtype=object)
    for i in range(4):
        for j in range(4):
            temp_state[i] = temp_state[i] + i_rx[i, j] * state[j]
    state = temp_state
    
    return state

def simple_qaoa_circuit_cudaq(gamma, beta):
    """Implement a single layer of QAOA for a 2-qubit system (single edge) using CUDA-Q
    
    Args:
        gamma: Problem Hamiltonian angle
        beta: Mixer Hamiltonian angle
        
    Returns:
        Final state vector after QAOA circuit
    """
    @cudaq.kernel
    def qaoa_kernel(gamma: float, beta: float):
        # Allocate 2 qubits
        q0 = cudaq.qubit()
        q1 = cudaq.qubit()
        
        # Step 1: Apply Hadamard to both qubits (create superposition)
        h(q0)
        h(q1)
        
        # Step 2: Apply problem Hamiltonian for the edge (0,1)
        # exp(-i*gamma*Z⊗Z) can be implemented as: CNOT -> Rz(2*gamma) -> CNOT
        x.ctrl(q0, q1)  # CNOT
        rz(2.0 * gamma, q1)  # Rz(2*gamma)
        x.ctrl(q0, q1)  # CNOT
        
        # Step 3: Apply mixer Hamiltonian
        rx(2.0 * beta, q0)  # RX on first qubit
        rx(2.0 * beta, q1)  # RX on second qubit
    
    # Retrieve state vector
    state = cudaq.get_state(qaoa_kernel, gamma, beta)
    return np.array(state)

def analyze_qaoa_solution(state_vector):
    """Analyze the QAOA output to find the Max-Cut solution for a single edge"""
    # For a 2-qubit system with a single edge, the max cut is achieved when
    # the two qubits have different values (01 or 10)
    probabilities = np.abs(state_vector)**2
    
    print(f"State vector: {state_vector}")
    print(f"Probabilities: {probabilities}")
    
    # For a 2-qubit system
    bitstrings = ["00", "01", "10", "11"]
    for i, bitstring in enumerate(bitstrings):
        print(f"Probability of {bitstring}: {probabilities[i]:.6f}")
    
    # Find the most likely bitstring
    most_likely_idx = np.argmax(probabilities)
    most_likely = bitstrings[most_likely_idx]
    print(f"Most likely outcome: {most_likely}")
    
    # For a single edge, calculate the cut value (1 if different bits, 0 if same)
    cut_value = 1 if most_likely[0] != most_likely[1] else 0
    print(f"Cut value: {cut_value}")
    
    return most_likely, cut_value

def test_simple_qaoa():
    """Test a simple QAOA implementation for a 2-qubit system with a single edge"""
    print("\nTesting simple QAOA circuit (2 qubits, 1 edge)...")
    
    # Choose test parameters (optimal for a single edge would be gamma=pi/4, beta=pi/8)
    gamma = np.pi / 4  # Problem Hamiltonian angle
    beta = np.pi / 8   # Mixer Hamiltonian angle
    
    print(f"Parameters: gamma={gamma}, beta={beta}")
    
    # Run all three implementations
    result_np = simple_qaoa_circuit_numpy(gamma, beta, dtype=np.complex64)
    result_fp16 = simple_qaoa_circuit_fp16(gamma, beta)
    result_cudaq = simple_qaoa_circuit_cudaq(gamma, beta)
    
    # Convert FP16 results to complex64 for comparison
    result_fp16_complex = np.array([x.to_complex64() for x in result_fp16])
    
    # Calculate fidelity
    fid_np_cudaq = fidelity(result_np, result_cudaq)
    fid_fp16_np = fidelity(result_fp16_complex, result_np)
    fid_fp16_cudaq = fidelity(result_fp16_complex, result_cudaq)
    
    print("\nQAOA state vector comparison:")
    print("CUDA-Q Result:", result_cudaq)
    print("NumPy Result:", result_np)
    print("FP16 Result:", [str(x) for x in result_fp16])
    print(f"Fidelity between NumPy and CUDA-Q: {fid_np_cudaq:.6f}")
    print(f"Fidelity between FP16 and CUDA-Q: {fid_fp16_cudaq:.6f}")
    print(f"Fidelity between FP16 and NumPy: {fid_fp16_np:.6f}")
    
    # Analyze the solutions
    print("\nAnalyzing CUDA-Q solution:")
    analyze_qaoa_solution(result_cudaq)
    
    print("\nAnalyzing NumPy solution:")
    analyze_qaoa_solution(result_np)
    
    print("\nAnalyzing FP16 solution:")
    analyze_qaoa_solution(result_fp16_complex)

# ---------- Main Execution ----------

def main():
    num_tests = 100
    error_threshold = 0.95
    
    # Test RX gate
    print("Testing RX gate implementation...")
    for i in range(num_tests):
        theta = np.random.uniform(0, 2 * np.pi)
        
        # Test with |0⟩ state
        result_np = simulate_numpy_rx(theta, dtype=np.complex64)
        result_cudaq = simulate_cudaq_rx(theta)
        fid = fidelity(result_np, result_cudaq)
        
        # Test FP16 implementation
        result_fp16 = simulate_fp16_rx(theta)
        
        # Convert FP16 results to complex64 for fidelity comparison
        result_fp16_complex = np.array([x.to_complex64() for x in result_fp16])
        fid_fp16_np = fidelity(result_fp16_complex, result_np)
        fid_fp16_cudaq = fidelity(result_fp16_complex, result_cudaq)
        
        if fid_fp16_cudaq < error_threshold:
            print(f"RX gate (|0⟩): FP16 implementation failed for test {i+1} with theta = {theta}")
            print("")
            print("CUDA-Q Result:", result_cudaq)
            print("NumPy Result:", result_np)
            print("FP16 Result:", [str(x) for x in result_fp16])
            print(f"Fidelity between FP16 and CUDA-Q: {fid_fp16_cudaq:.6f}")
            print(f"Fidelity between NumPy and CUDA-Q: {fid:.6f}")
            print(f"Fidelity between FP16 and NumPy: {fid_fp16_np:.6f}")
    
    # Test H gate
    print("\nTesting H gate implementation...")
    
    # Test with |0⟩ state
    result_np = simulate_numpy_h(dtype=np.complex64)
    result_cudaq = simulate_cudaq_h()
    fid = fidelity(result_np, result_cudaq)
    
    # Test FP16 implementation
    result_fp16 = simulate_fp16_h()
    
    # Convert FP16 results to complex64 for fidelity comparison
    result_fp16_complex = np.array([x.to_complex64() for x in result_fp16])
    fid_fp16_np = fidelity(result_fp16_complex, result_np)
    fid_fp16_cudaq = fidelity(result_fp16_complex, result_cudaq)
    
    print("H gate results (|0⟩ input):")
    print("CUDA-Q Result:", result_cudaq)
    print("NumPy Result:", result_np)
    print("FP16 Result:", [str(x) for x in result_fp16])
    print(f"Fidelity between FP16 and CUDA-Q: {fid_fp16_cudaq:.6f}")
    print(f"Fidelity between NumPy and CUDA-Q: {fid:.6f}")
    print(f"Fidelity between FP16 and NumPy: {fid_fp16_np:.6f}")
    
    # Test RZ gate
    print("\nTesting RZ gate implementation...")
    for i in range(num_tests):
        theta = np.random.uniform(0, 2 * np.pi)
        
        # Test with |0⟩ state
        result_np = simulate_numpy_rz(theta, dtype=np.complex64)
        result_cudaq = simulate_cudaq_rz(theta)
        fid = fidelity(result_np, result_cudaq)
        
        # Test FP16 implementation
        result_fp16 = simulate_fp16_rz(theta)
        
        # Convert FP16 results to complex64 for fidelity comparison
        result_fp16_complex = np.array([x.to_complex64() for x in result_fp16])
        fid_fp16_np = fidelity(result_fp16_complex, result_np)
        fid_fp16_cudaq = fidelity(result_fp16_complex, result_cudaq)
        
        if fid_fp16_cudaq < error_threshold:
            print(f"RZ gate (|0⟩): FP16 implementation failed for test {i+1} with theta = {theta}")
            print("")
            print("CUDA-Q Result:", result_cudaq)
            print("NumPy Result:", result_np)
            print("FP16 Result:", [str(x) for x in result_fp16])
            print(f"Fidelity between FP16 and CUDA-Q: {fid_fp16_cudaq:.6f}")
            print(f"Fidelity between NumPy and CUDA-Q: {fid:.6f}")
            print(f"Fidelity between FP16 and NumPy: {fid_fp16_np:.6f}")

    # Test CZ gate
    test_cz_gate()

    # Test tensor product
    test_tensor_product()

    # Test CNOT gate
    test_cnot_gate()
    
    # Test ZZ interaction
    test_zz_interaction()

    # Test simple QAOA circuit
    test_simple_qaoa()

if __name__ == "__main__":
    main()

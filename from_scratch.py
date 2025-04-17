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

# ---------- FP16 Complex Implementation ----------
class ComplexFP16:
    """A complex number represented by two fp16 values (real and imaginary parts)"""
    def __init__(self, real, imag):
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
        [ComplexFP16(cos, 0), ComplexFP16(0, -sin)],
        [ComplexFP16(0, -sin), ComplexFP16(cos, 0)]
    ])

def simulate_fp16_rx(theta):
    """Apply RX to |0⟩ using FP16 complex numbers and return resulting state vector"""
    input_state = np.array([ComplexFP16(1.0, 0.0), ComplexFP16(0.0, 0.0)])
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
        [ComplexFP16(h_factor, 0.0), ComplexFP16(h_factor, 0.0)],
        [ComplexFP16(h_factor, 0.0), ComplexFP16(-h_factor, 0.0)]
    ])

def simulate_fp16_h():
    """Apply H to |0⟩ using FP16 complex numbers and return resulting state vector"""
    input_state = np.array([ComplexFP16(1.0, 0.0), ComplexFP16(0.0, 0.0)])
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
        [ComplexFP16(cos_neg, sin_neg), ComplexFP16(0.0, 0.0)],
        [ComplexFP16(0.0, 0.0), ComplexFP16(cos_pos, sin_pos)]
    ])

def simulate_fp16_rz(theta):
    """Apply RZ to |0⟩ using FP16 complex numbers and return resulting state vector"""
    input_state = np.array([ComplexFP16(1.0, 0.0), ComplexFP16(0.0, 0.0)])
    rz = rz_gate_fp16(theta)
    
    # Manual matrix multiplication with FP16 complex numbers
    result = np.array([
        rz[0, 0] * input_state[0] + rz[0, 1] * input_state[1],
        rz[1, 0] * input_state[0] + rz[1, 1] * input_state[1]
    ])
    
    return result

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
        [ComplexFP16(1.0, 0.0), ComplexFP16(0.0, 0.0), ComplexFP16(0.0, 0.0), ComplexFP16(0.0, 0.0)],
        [ComplexFP16(0.0, 0.0), ComplexFP16(1.0, 0.0), ComplexFP16(0.0, 0.0), ComplexFP16(0.0, 0.0)],
        [ComplexFP16(0.0, 0.0), ComplexFP16(0.0, 0.0), ComplexFP16(1.0, 0.0), ComplexFP16(0.0, 0.0)],
        [ComplexFP16(0.0, 0.0), ComplexFP16(0.0, 0.0), ComplexFP16(0.0, 0.0), ComplexFP16(-1.0, 0.0)]
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
            ComplexFP16(1.0, 0.0), 
            ComplexFP16(0.0, 0.0), 
            ComplexFP16(0.0, 0.0), 
            ComplexFP16(0.0, 0.0)
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
        ComplexFP16(0.0, 0.0), 
        ComplexFP16(0.0, 0.0), 
        ComplexFP16(0.0, 0.0), 
        ComplexFP16(1.0, 0.0)
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
        [ComplexFP16(1.0, 0.0), ComplexFP16(0.0, 0.0)],
        [ComplexFP16(0.0, 0.0), ComplexFP16(1.0, 0.0)]
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
        ComplexFP16(1.0, 0.0),
        ComplexFP16(0.0, 0.0),
        ComplexFP16(0.0, 0.0),
        ComplexFP16(0.0, 0.0)
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
        [ComplexFP16(1.0, 0.0), ComplexFP16(0.0, 0.0), ComplexFP16(0.0, 0.0), ComplexFP16(0.0, 0.0)],
        [ComplexFP16(0.0, 0.0), ComplexFP16(1.0, 0.0), ComplexFP16(0.0, 0.0), ComplexFP16(0.0, 0.0)],
        [ComplexFP16(0.0, 0.0), ComplexFP16(0.0, 0.0), ComplexFP16(0.0, 0.0), ComplexFP16(1.0, 0.0)],
        [ComplexFP16(0.0, 0.0), ComplexFP16(0.0, 0.0), ComplexFP16(1.0, 0.0), ComplexFP16(0.0, 0.0)]
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
            ComplexFP16(1.0, 0.0), 
            ComplexFP16(0.0, 0.0), 
            ComplexFP16(0.0, 0.0), 
            ComplexFP16(0.0, 0.0)
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
        ComplexFP16(0.0, 0.0), 
        ComplexFP16(0.0, 0.0), 
        ComplexFP16(1.0, 0.0), 
        ComplexFP16(0.0, 0.0)
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
        [ComplexFP16(cos_neg, sin_neg), ComplexFP16(0.0, 0.0), ComplexFP16(0.0, 0.0), ComplexFP16(0.0, 0.0)],
        [ComplexFP16(0.0, 0.0), ComplexFP16(cos_pos, sin_pos), ComplexFP16(0.0, 0.0), ComplexFP16(0.0, 0.0)],
        [ComplexFP16(0.0, 0.0), ComplexFP16(0.0, 0.0), ComplexFP16(cos_pos, sin_pos), ComplexFP16(0.0, 0.0)],
        [ComplexFP16(0.0, 0.0), ComplexFP16(0.0, 0.0), ComplexFP16(0.0, 0.0), ComplexFP16(cos_neg, sin_neg)]
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
            ComplexFP16(1.0, 0.0), 
            ComplexFP16(0.0, 0.0), 
            ComplexFP16(0.0, 0.0), 
            ComplexFP16(0.0, 0.0)
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
        ComplexFP16(0.5, 0.0), 
        ComplexFP16(0.5, 0.0), 
        ComplexFP16(0.5, 0.0), 
        ComplexFP16(0.5, 0.0)
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

# ---------- QAOA Circuit Components ----------

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

def simulate_cudaq_qaoa_layer(gamma, beta, edge_list, num_qubits):
    """Use CUDA-Q to apply one layer of QAOA with given parameters
    
    Args:
        gamma: Problem Hamiltonian parameter
        beta: Mixer Hamiltonian parameter
        edge_list: List of edges as [source, target] pairs
        num_qubits: Number of qubits
        
    Returns:
        State vector after applying one layer of QAOA
    """
    @cudaq.kernel
    def qaoa_layer(gamma: float, beta: float, edges_src: List[int], edges_tgt: List[int]):
        qubits = cudaq.qvector(num_qubits)
        
        # Initialize in |+⟩ state
        h(qubits)
        
        # Apply problem Hamiltonian (ZZ rotations)
        for edge_idx in range(len(edges_src)):
            qubit_u = edges_src[edge_idx]
            qubit_v = edges_tgt[edge_idx]
            
            # Apply exp(-i*gamma*Z_u*Z_v) using CNOT-RZ-CNOT
            x.ctrl(qubits[qubit_u], qubits[qubit_v])
            rz(2.0 * gamma, qubits[qubit_v])
            x.ctrl(qubits[qubit_u], qubits[qubit_v])
        
        # Apply mixer Hamiltonian (X rotations)
        for qubit_idx in range(num_qubits):
            rx(2.0 * beta, qubits[qubit_idx])
    
    # Extract the edges into source and target lists
    edges_src = [edge[0] for edge in edge_list]
    edges_tgt = [edge[1] for edge in edge_list]
    
    # Run the kernel and get the state vector
    state = cudaq.get_state(qaoa_layer, gamma, beta, edges_src, edges_tgt)
    return np.array(state)

def test_qaoa_components():
    """Test the QAOA component implementations"""
    # Test parameters
    gamma = np.pi / 8  # Problem Hamiltonian angle
    beta = np.pi / 4   # Mixer Hamiltonian angle
    
    # Test on a small graph (triangle)
    num_qubits = 3
    edges = [[0, 1], [1, 2], [0, 2]]
    
    print("\nTesting QAOA components implementation...")
    
    # Create initial state |+++⟩ using FP16
    initial_state_fp16 = np.zeros(2**num_qubits, dtype=object)
    h_factor = np.float16(0.7071067811865476)  # 1/sqrt(2)
    for i in range(2**num_qubits):
        initial_state_fp16[i] = ComplexFP16(h_factor**num_qubits, 0.0)
    
    # Create FP16 QAOA kernels
    problem_kernel_fp16 = qaoa_problem_kernel_fp16(gamma, edges, num_qubits)
    mixer_kernel_fp16 = qaoa_mixer_kernel_fp16(beta, num_qubits)
    
    # Apply QAOA using FP16
    result_fp16 = qaoa_apply_fp16(initial_state_fp16, problem_kernel_fp16, mixer_kernel_fp16)
    
    # Apply QAOA using CUDA-Q
    result_cudaq = simulate_cudaq_qaoa_layer(gamma, beta, edges, num_qubits)
    
    # Convert FP16 results to complex64 for comparison
    result_fp16_complex = np.array([x.to_complex64() for x in result_fp16])
    
    # Calculate fidelity
    fid = fidelity(result_fp16_complex, result_cudaq)
    
    print(f"\nQAOA with gamma={gamma}, beta={beta} on triangle graph:")
    print(f"Fidelity between FP16 and CUDA-Q implementations: {fid:.6f}")
    
    # Print a few state amplitudes for comparison
    print("\nSample state amplitudes comparison:")
    print("State | CUDA-Q | FP16")
    for i in range(min(8, len(result_cudaq))):
        bin_rep = format(i, f'0{num_qubits}b')
        print(f"|{bin_rep}⟩ | {result_cudaq[i]:.6f} | {result_fp16_complex[i]:.6f}")
    
    return fid > 0.99  # Return True if fidelity is high enough

# ---------- Main Execution ----------

# Add QAOA implementation using FP16
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

def test_maxcut_qaoa():
    """Test the MaxCut QAOA implementation with FP16"""
    print("\nTesting MaxCut QAOA implementation with FP16...")
    
    # Create a simple triangle graph
    nodes = [0, 1, 2]
    edges = [[0, 1], [1, 2], [0, 2]]
    
    # Create the QAOA solver
    qaoa = MaxCutQAOA_FP16(nodes, edges, layer_count=1, seed=42)
    
    # Try a single set of parameters
    gamma = np.pi / 8
    beta = np.pi / 4
    parameters = [gamma, beta]
    
    # Apply the QAOA circuit
    state = qaoa.apply_qaoa_circuit(parameters)
    
    # Print the state vector
    print("QAOA state vector:")
    state_complex = np.array([x.to_complex64() for x in state])
    for i, amp in enumerate(state_complex):
        bitstring = format(i, f'0{qaoa.qubit_count}b')
        # print(f"|{bitstring}⟩: {amp}")
    
    # Evaluate the expectation value
    expectation = qaoa.evaluate_expectation(state)
    print(f"Expectation value: {expectation.real}")
    
    # Now create the CUDA-Q circuit for comparison
    @cudaq.kernel
    def qaoa_kernel(gamma: float, beta: float, edges_src: List[int], edges_tgt: List[int]):
        qubits = cudaq.qvector(len(nodes))
        
        # Initialize in |+⟩ state
        h(qubits)
        
        # Apply problem Hamiltonian
        for edge_idx in range(len(edges_src)):
            qubit_u = edges_src[edge_idx]
            qubit_v = edges_tgt[edge_idx]
            
            # Apply exp(-i*gamma*Z_u*Z_v) using CNOT-RZ-CNOT
            x.ctrl(qubits[qubit_u], qubits[qubit_v])
            rz(2.0 * gamma, qubits[qubit_v])
            x.ctrl(qubits[qubit_u], qubits[qubit_v])
        
        # Apply mixer Hamiltonian
        for qubit_idx in range(len(nodes)):
            rx(2.0 * beta, qubits[qubit_idx])
    
    # Extract edges for CUDA-Q
    edges_src = [edge[0] for edge in edges]
    edges_tgt = [edge[1] for edge in edges]
    
    # Get the CUDA-Q state vector
    cudaq_state = cudaq.get_state(qaoa_kernel, gamma, beta, edges_src, edges_tgt)
    
    # print("CUDA-Q state vector:")
    # for i, amp in enumerate(cudaq_state):
    #     bitstring = format(i, f'0{qaoa.qubit_count}b')
    #     # print(f"|{bitstring}⟩: {amp}")
    
    # Compare the state vectors
    fp16_state_complex = np.array([x.to_complex64() for x in state])
    fid = fidelity(fp16_state_complex, cudaq_state)
    
    print(f"Fidelity between FP16 and CUDA-Q implementations: {fid:.6f}")
    
    # Now run a small optimization
    print("\nRunning small optimization...")
    optimal_energy, optimal_parameters = qaoa.optimize(iterations=10)
    
    print(f"Optimal energy: {optimal_energy}")
    print(f"Optimal parameters: {optimal_parameters}")
    
    # Sample from the optimized circuit
    counts = qaoa.sample(optimal_parameters, shots=100)
    
    print("Measurement counts:")
    for bitstring, count in counts.items():
        print(f"|{bitstring}⟩: {count}")
    
    # Get the max cut
    max_cut, cut_value = qaoa.get_max_cut(counts)
    
    print(f"Max cut: {max_cut}")
    print(f"Cut value: {cut_value}")
    
    return fid > 0.99

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
    
    # Test QAOA components
    test_qaoa_components()

    # Test the MaxCut QAOA implementation
    test_maxcut_qaoa()

if __name__ == "__main__":
    main()

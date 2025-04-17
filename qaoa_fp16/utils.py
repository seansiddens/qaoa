import numpy as np

def fidelity(a, b):
    """Compute the fidelity (|⟨a|b⟩|^2) between two state vectors"""
    return np.abs(np.vdot(a, b))**2

def get_norm_deviation(state):
    norm = np.sqrt(np.sum(np.abs(state)**2))
    deviation = abs(norm - 1.0)
    return deviation
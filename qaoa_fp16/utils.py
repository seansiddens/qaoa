import numpy as np

def fidelity(a, b):
    """Compute the fidelity (|⟨a|b⟩|^2) between two state vectors"""
    return np.abs(np.vdot(a, b))**2
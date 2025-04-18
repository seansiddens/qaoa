import numpy as np

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

    def conjugate(self):
        return ComplexFP16(self.real, -self.imag)
    
    def to_complex64(self):
        """Convert to numpy complex64 for compatibility with other functions"""
        return np.complex64(self.real.astype(np.float32) + 1j * self.imag.astype(np.float32))

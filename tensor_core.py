"""
Elastic tensor operation kernel
"""

import numpy as np
from typing import Tuple
from constants import PI, TWOPI

class TensorOperations:
    """Operações fundamentais com tensor de 4ª ordem"""
    
    @staticmethod
    def voigt_to_tensor(S: np.ndarray) -> np.ndarray:
        """
        Converte matriz de compliance 6x6 para tensor completo Sijkl
        """
        S4 = np.zeros((3, 3, 3, 3))
        
        voigt_map = [
            (0, 0),  # 11 → 00
            (1, 1),  # 22 → 11
            (2, 2),  # 33 → 22
            (1, 2),  # 23 → 12
            (0, 2),  # 13 → 02
            (0, 1)   # 12 → 01
        ]
        
        for i in range(6):
            a, b = voigt_map[i]
            for j in range(6):
                c, d = voigt_map[j]
                
                factor = 1.0
                if i >= 3:
                    factor *= 0.5
                if j >= 3:
                    factor *= 0.5
                
                value = S[i, j] * factor
                
                S4[a, b, c, d] = value
                S4[b, a, c, d] = value
                S4[a, b, d, c] = value
                S4[b, a, d, c] = value
        
        return S4
    
    @staticmethod
    def build_orthonormal_basis(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constrói base ortonormal estável para qualquer direção n
        """
        n = n / np.linalg.norm(n)
        
        A = np.zeros((3, 3))
        A[0, :] = n
        
        if abs(n[0]) < 0.9:
            A[1, :] = [1.0, 0.0, 0.0]
            A[2, :] = [0.0, 1.0, 0.0]
        elif abs(n[1]) < 0.9:
            A[1, :] = [0.0, 1.0, 0.0]
            A[2, :] = [0.0, 0.0, 1.0]
        else:
            A[1, :] = [1.0, 0.0, 0.0]
            A[2, :] = [0.0, 0.0, 1.0]
        
        Q, R = np.linalg.qr(A.T)
        
        u = Q[:, 1]
        v = Q[:, 2]
        
        v = v - np.dot(v, n) * n
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-12:
            v = v / v_norm
        
        return u, v
    
    @staticmethod
    def tensor_contraction(S4: np.ndarray, a: np.ndarray, b: np.ndarray, 
                          c: np.ndarray, d: np.ndarray) -> float:
        """Contração tensorial usando np.einsum"""
        return np.einsum('i,j,k,l,ijkl', a, b, c, d, S4, optimize=True)
    
    @staticmethod
    def direction_vector(theta: float, phi: float) -> np.ndarray:
        """Vetor direção a partir de ângulos esféricos"""
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.array([x, y, z])
    
    @staticmethod
    def angles_from_vector(n: np.ndarray) -> Tuple[float, float]:
        """Ângulos esféricos a partir de vetor direção"""
        n = n / np.linalg.norm(n)
        theta = np.arccos(n[2])
        phi = np.arctan2(n[1], n[0])
        if phi < 0:
            phi += TWOPI
        return theta, phi

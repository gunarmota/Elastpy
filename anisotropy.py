"""
Anisotropy indices
"""

import numpy as np
from typing import Tuple

class AnisotropyCalculator:
    """Calcula diversos índices de anisotropia"""
    
    @staticmethod
    def universal(B_V: float, G_V: float, B_R: float, G_R: float) -> Tuple[float, float]:
        """Índice universal A_U e anisotropia de compressibilidade"""
        A_U = 5 * G_V / G_R + B_V / B_R - 6
        A_comp = (B_V - B_R) / (B_V + B_R) * 100
        return A_U, A_comp
    
    @staticmethod
    def shear(C: np.ndarray) -> float:
        """Anisotropia de cisalhamento"""
        try:
            A_shear = 2*C[3,3] / (C[0,0] - C[0,1])
        except:
            A_shear = 1.0
        return A_shear
    
    @staticmethod
    def zener(C: np.ndarray, symmetry: str) -> float:
        """Anisotropia de Zener"""
        if symmetry == "Cubic":
            return 2*C[3,3] / (C[0,0] - C[0,1])
        else:
            A1 = 2*C[3,3] / (C[0,0] + C[1,1] - 2*C[0,1])
            A2 = 2*C[4,4] / (C[1,1] + C[2,2] - 2*C[1,2])
            A3 = 2*C[5,5] / (C[0,0] + C[2,2] - 2*C[0,2])
            return np.mean([A1, A2, A3])
    
    @staticmethod
    def chung(B_V: float, G_V: float, B_R: float, G_R: float) -> float:
        """Anisotropia de Chung"""
        A_B = (B_V - B_R) / (B_V + B_R)
        A_G = (G_V - G_R) / (G_V + G_R)
        return np.sqrt(A_B**2 + A_G**2)
    
    @staticmethod
    def young(E_max: float, E_min: float) -> float:
        """Anisotropia do módulo de Young"""
        if E_min > 0:
            return E_max / E_min
        return 1.0

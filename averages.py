"""
Voigt-Reuss-Hill average calculation
"""

import numpy as np
from typing import Tuple

class AverageCalculator:
    """Calcula médias elásticas pelos métodos de Voigt, Reuss e Hill"""
    
    @staticmethod
    def voigt(C: np.ndarray) -> Tuple[float, float]:
        """Médias de Voigt"""
        B_V = (C[0,0] + C[1,1] + C[2,2] + 2*(C[0,1] + C[0,2] + C[1,2])) / 9.0
        G_V = (C[0,0] + C[1,1] + C[2,2] - (C[0,1] + C[0,2] + C[1,2]) +
               3*(C[3,3] + C[4,4] + C[5,5])) / 15.0
        return B_V, G_V
    
    @staticmethod
    def reuss(S: np.ndarray) -> Tuple[float, float]:
        """Médias de Reuss"""
        s11, s22, s33 = S[0,0], S[1,1], S[2,2]
        s12, s13, s23 = S[0,1], S[0,2], S[1,2]
        B_R = 1.0 / (s11 + s22 + s33 + 2*(s12 + s13 + s23))
        
        s44, s55, s66 = S[3,3], S[4,4], S[5,5]
        G_R = 15.0 / (4*(s11 + s22 + s33 - (s12 + s13 + s23)) + 
                      3*(s44 + s55 + s66))
        return B_R, G_R
    
    @staticmethod
    def hill(B_V: float, G_V: float, B_R: float, G_R: float) -> Tuple[float, float, float, float]:
        """Médias de Hill e propriedades derivadas"""
        B_H = (B_V + B_R) / 2.0
        G_H = (G_V + G_R) / 2.0
        
        E_H = 9 * B_H * G_H / (3 * B_H + G_H)
        nu_H = (3 * B_H - 2 * G_H) / (2 * (3 * B_H + G_H))
        
        return B_H, G_H, E_H, nu_H

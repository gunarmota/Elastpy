"""
Thermoacoustic properties
"""

import numpy as np
from typing import Tuple

class ThermoAcousticProperties:
    """Calcula velocidades do som e temperatura de Debye"""
    
    @staticmethod
    def sound_velocities(B_H: float, G_H: float, density: float) -> Tuple[float, float, float, float]:
        """Velocidades do som em km/s"""
        if density <= 0:
            density = 1.0
        
        B_H_Pa = B_H * 1e9
        G_H_Pa = G_H * 1e9
        
        vp = np.sqrt((B_H_Pa + 4*G_H_Pa/3) / density) / 1000.0
        vs = np.sqrt(G_H_Pa / density) / 1000.0
        
        if vp > 0 and vs > 0:
            vm = 1 / np.cbrt((1/3) * (1/vp**3 + 2/vs**3))
        else:
            vm = 0
        
        return vp, vs, vm, vp/vs if vs > 0 else 0
    
    @staticmethod
    def debye_temperature(vm: float, n_atoms: float, volume: float) -> float:
        """Temperatura de Debye em Kelvin"""
        from scipy.constants import h, k
        
        if vm <= 0:
            return 0.0
        
        h_planck = h
        k_B = k
        
        n_per_vol = n_atoms / (volume * 1e-30)
        theta_D = (h_planck / k_B) * (3 * n_per_vol / (4 * np.pi))**(1/3) * vm * 1000
        
        return theta_D
    
    @staticmethod
    def hardness(B_H: float, G_H: float) -> float:
        """Dureza estimada (modelo de Chen)"""
        k = G_H / B_H
        if k > 0.6:
            H = 0.151 * G_H
        else:
            H = 0.1769 * G_H - 2.899
        
        return max(0, H)

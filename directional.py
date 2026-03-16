"""
Directional properties
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks, savgol_filter
from typing import Tuple
from constants import TWOPI
from tensor_core import TensorOperations

class DirectionalProperties:
    """Calcula propriedades direcionais usando metodologia ElastPy melhorada"""
    
    def __init__(self, C: np.ndarray, S: np.ndarray, S4: np.ndarray):
        self.C = C
        self.S = S
        self.S4 = S4
        self.tensor_ops = TensorOperations()
    
    def young_modulus(self, n: np.ndarray) -> float:
        """Módulo de Young na direção n"""
        n = n / np.linalg.norm(n)
        E_inv = self.tensor_ops.tensor_contraction(self.S4, n, n, n, n)
        return 1.0 / E_inv
    
    def linear_compressibility(self, n: np.ndarray) -> float:
        """Compressibilidade linear na direção n"""
        S = self.S
        n1, n2, n3 = n
        n1, n2, n3 = n1*n1, n2*n2, n3*n3
        
        beta = ((S[0,0] + S[0,1] + S[0,2])*n1 +
                (S[0,1] + S[1,1] + S[1,2])*n2 +
                (S[0,2] + S[1,2] + S[2,2])*n3 +
                (S[0,5] + S[1,5] + S[2,5])*2*n1*n2 +
                (S[0,4] + S[1,4] + S[2,4])*2*n1*n3 +
                (S[0,3] + S[1,3] + S[2,3])*2*n2*n3)
        
        return beta * 1000.0
    
    def shear_modulus_improved(self, n: np.ndarray, nsteps_chi: int = 180) -> Tuple[float, float]:
        """Módulo de cisalhamento com χ ∈ [0, 2π] - método melhorado"""
        n = n / np.linalg.norm(n)
        u, v = self.tensor_ops.build_orthonormal_basis(n)
        
        def G_of_chi(chi):
            m = np.cos(chi) * u + np.sin(chi) * v
            m = m - np.dot(m, n) * n
            m_norm = np.linalg.norm(m)
            if m_norm > 1e-12:
                m = m / m_norm
            else:
                return 0.0
            
            G_inv = 4.0 * self.tensor_ops.tensor_contraction(self.S4, n, m, n, m)
            return 1.0 / G_inv if G_inv > 1e-12 else 0.0
        
        chis = np.linspace(0, TWOPI, nsteps_chi)
        G_vals = np.array([G_of_chi(chi) for chi in chis])
        
        if len(G_vals) > 10:
            window = min(11, len(G_vals) // 10)
            if window % 2 == 0:
                window += 1
            G_smooth = savgol_filter(G_vals, window, 3)
        else:
            G_smooth = G_vals
        
        peaks, _ = find_peaks(G_smooth)
        valleys, _ = find_peaks(-G_smooth)
        
        candidates_max = []
        candidates_min = []
        delta = TWOPI / nsteps_chi
        
        for idx in peaks:
            chi_guess = chis[idx]
            res = minimize_scalar(
                lambda x: -G_of_chi(x),
                bounds=(max(0, chi_guess - delta), 
                       min(TWOPI, chi_guess + delta)),
                method='bounded'
            )
            candidates_max.append(-res.fun)
        
        for idx in valleys:
            chi_guess = chis[idx]
            res = minimize_scalar(
                G_of_chi,
                bounds=(max(0, chi_guess - delta), 
                       min(TWOPI, chi_guess + delta)),
                method='bounded'
            )
            candidates_min.append(res.fun)
        
        G_min = min(candidates_min) if candidates_min else G_vals.min()
        G_max = max(candidates_max) if candidates_max else G_vals.max()
        
        return G_min, G_max
    
    def poisson_ratio_improved(self, n: np.ndarray, nsteps_chi: int = 180) -> Tuple[float, float]:
        """Coeficiente de Poisson com χ ∈ [0,2π]"""
        n = n / np.linalg.norm(n)
        u, v = self.tensor_ops.build_orthonormal_basis(n)
        
        E_inv = self.tensor_ops.tensor_contraction(self.S4, n, n, n, n)
        E = 1.0 / E_inv
        
        def nu_of_chi(chi):
            m = np.cos(chi) * u + np.sin(chi) * v
            m = m - np.dot(m, n) * n
            m_norm = np.linalg.norm(m)
            if m_norm > 1e-12:
                m = m / m_norm
            else:
                return 0.0
            
            val = self.tensor_ops.tensor_contraction(self.S4, m, m, n, n)
            return -E * val
        
        chis = np.linspace(0, TWOPI, nsteps_chi)
        nu_vals = np.array([nu_of_chi(chi) for chi in chis])
        
        if len(nu_vals) > 10:
            window = min(11, len(nu_vals) // 10)
            if window % 2 == 0:
                window += 1
            nu_smooth = savgol_filter(nu_vals, window, 3)
        else:
            nu_smooth = nu_vals
        
        peaks, _ = find_peaks(nu_smooth)
        valleys, _ = find_peaks(-nu_smooth)
        
        candidates_max = []
        candidates_min = []
        delta = TWOPI / nsteps_chi
        
        for idx in peaks:
            chi_guess = chis[idx]
            res = minimize_scalar(
                lambda x: -nu_of_chi(x),
                bounds=(max(0, chi_guess - delta), 
                       min(TWOPI, chi_guess + delta)),
                method='bounded'
            )
            candidates_max.append(-res.fun)
        
        for idx in valleys:
            chi_guess = chis[idx]
            res = minimize_scalar(
                nu_of_chi,
                bounds=(max(0, chi_guess - delta), 
                       min(TWOPI, chi_guess + delta)),
                method='bounded'
            )
            candidates_min.append(res.fun)
        
        nu_min = min(candidates_min) if candidates_min else nu_vals.min()
        nu_max = max(candidates_max) if candidates_max else nu_vals.max()
        
        return nu_min, nu_max

"""
Crystal symmetry identification
"""

import numpy as np
from typing import Tuple

class SymmetryIdentifier:
    """Identifica a simetria do cristal baseado na matriz C"""
    
    @staticmethod
    def identify(C: np.ndarray, tol: float = 1e-6) -> str:
        zeros = np.abs(C) < tol
        
        nz_upper = np.sum(~zeros[:3, 3:6])
        nz_lower = np.sum(~zeros[3:6, :3])
        nz_coupling = nz_upper + nz_lower
    
        # Cúbico
        if (np.abs(C[0,0] - C[1,1]) < tol and 
            np.abs(C[0,0] - C[2,2]) < tol and
            np.abs(C[3,3] - C[4,4]) < tol and
            np.abs(C[3,3] - C[5,5]) < tol and
            np.abs(C[0,1] - C[0,2]) < tol and
            np.abs(C[0,1] - C[1,2]) < tol and
            np.all(zeros[0,3:6]) and np.all(zeros[1,3:6]) and 
            np.all(zeros[2,3:6]) and np.all(zeros[3:6,0:3])):
            return "Cubic"
        
        # Hexagonal
        if (np.abs(C[0,0] - C[1,1]) < tol and
            np.abs(C[0,2] - C[1,2]) < tol and
            np.abs(C[3,3] - C[4,4]) < tol and
            np.abs(C[5,5] - (C[0,0] - C[0,1])/2) < tol and
            np.all(zeros[0,3:6]) and np.all(zeros[1,3:6]) and 
            np.all(zeros[2,3:6]) and np.all(zeros[3:6,0:3])):
            return "Hexagonal"
        
        # Tetragonal
        if (np.abs(C[0,0] - C[1,1]) < tol and
            np.abs(C[0,2] - C[1,2]) < tol and
            np.abs(C[3,3] - C[4,4]) < tol and
            np.all(zeros[0,3:6]) and np.all(zeros[1,3:6]) and 
            np.all(zeros[2,3:6]) and np.all(zeros[3:6,0:3])):
            return "Tetragonal"
        
        # Ortorrômbico
        if (np.all(zeros[0,3:6]) and np.all(zeros[1,3:6]) and 
            np.all(zeros[2,3:6]) and np.all(zeros[3:6,0:3]) and
            np.all(zeros[3,4:6]) and np.all(zeros[4,5:6]) and
            np.abs(C[3,4]) < tol and np.abs(C[3,5]) < tol and 
            np.abs(C[4,5]) < tol):
            return "Orthorhombic"
        
        # Trigonal / Romboédrico
        if (np.abs(C[0,0] - C[1,1]) < tol and
            np.abs(C[0,2] - C[1,2]) < tol and
            np.abs(C[3,3] - C[4,4]) < tol):
            
            if (np.abs(C[0,3] + C[1,3]) < tol and
                np.abs(C[0,4] - C[1,4]) < tol):
                return "Rhombohedral (Class I)"
            elif (np.abs(C[0,3]) < tol and 
                  np.abs(C[1,3]) < tol and
                  np.abs(C[5,5] - C[4,4]) < tol):
                return "Rhombohedral (Class II)"
            else:
                return "Rhombohedral"
        
        # Monoclínico
        if nz_coupling > 0:
            mono_patterns = [
                (~zeros[0,4] and ~zeros[1,4] and ~zeros[2,4] and ~zeros[3,5]),
                (~zeros[0,3] and ~zeros[1,3] and ~zeros[2,3] and ~zeros[4,5]),
                (~zeros[0,5] and ~zeros[1,5] and ~zeros[2,5] and ~zeros[3,4])
            ]
            
            if any(mono_patterns):
                return "Monoclinic"
        
        # Triclínico
        if nz_coupling > 3:
            return "Triclinic"
        
        return "Unknown"
    
    @staticmethod
    def map_to_irreducible_zone(n: np.ndarray, symmetry: str) -> np.ndarray:
        """Mapeia direção para zona irredutível baseada na simetria"""
        from .tensor_core import TensorOperations
        
        n = n / np.linalg.norm(n)
        
        if symmetry == "Cubic":
            n_red = np.abs(n)
            n_red = np.sort(n_red)[::-1]
            return n_red / np.linalg.norm(n_red)
        
        elif symmetry == "Hexagonal":
            theta, phi = TensorOperations.angles_from_vector(n)
            phi = phi % (np.pi/3)
            return TensorOperations.direction_vector(theta, phi)
        
        elif symmetry in ["Tetragonal", "Orthorhombic"]:
            return np.abs(n) / np.linalg.norm(np.abs(n))
        
        return n

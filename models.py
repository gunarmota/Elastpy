"""
Data classes for elastic properties
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class ElasticProperties:
    """Classe para armazenar propriedades elásticas calculadas"""
    # Propriedades médias (Voigt-Reuss-Hill)
    B_V: float = 0.0
    B_R: float = 0.0
    B_H: float = 0.0
    G_V: float = 0.0
    G_R: float = 0.0
    G_H: float = 0.0
    E_V: float = 0.0
    E_R: float = 0.0
    E_H: float = 0.0
    nu_V: float = 0.0
    nu_R: float = 0.0
    nu_H: float = 0.0
    
    # Razões e anisotropia
    Pugh_ratio: float = 0.0
    A_U: float = 0.0
    A_comp: float = 0.0
    A_shear: float = 0.0
    A_young: float = 0.0
    
    # Valores extremos e direções
    E_max: float = 0.0
    E_min: float = 0.0
    E_anisotropy: float = 0.0
    E_max_dir: np.ndarray = None
    E_min_dir: np.ndarray = None
    E_max_angles: Tuple[float, float] = (0.0, 0.0)
    E_min_angles: Tuple[float, float] = (0.0, 0.0)
    
    G_max: float = 0.0
    G_min: float = 0.0
    G_anisotropy: float = 0.0
    G_max_dir: np.ndarray = None
    G_min_dir: np.ndarray = None
    G_max_angles: Tuple[float, float] = (0.0, 0.0)
    G_min_angles: Tuple[float, float] = (0.0, 0.0)
    
    beta_max: float = 0.0
    beta_min: float = 0.0
    beta_anisotropy: float = 0.0
    beta_max_dir: np.ndarray = None
    beta_min_dir: np.ndarray = None
    beta_max_angles: Tuple[float, float] = (0.0, 0.0)
    beta_min_angles: Tuple[float, float] = (0.0, 0.0)
    
    nu_max: float = 0.0
    nu_min: float = 0.0
    nu_anisotropy: float = 0.0
    nu_max_dir: np.ndarray = None
    nu_min_dir: np.ndarray = None
    nu_max_angles: Tuple[float, float] = (0.0, 0.0)
    nu_min_angles: Tuple[float, float] = (0.0, 0.0)
    
    # Velocidades do som (km/s)
    vp: float = 0.0
    vs: float = 0.0
    vm: float = 0.0
    vp_vs_ratio: float = 0.0
    
    # Temperatura de Debye (K)
    Debye_temp: float = 0.0
    
    # Eigenvalues
    eigenvalues: np.ndarray = None
    
    # Índices de anisotropia universais
    Zener_anisotropy: float = 0.0
    Chung_anisotropy: float = 0.0
    
    # Dureza (modelo de Chen)
    hardness: float = 0.0

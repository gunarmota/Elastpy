"""
Classe principal ElasticTensor
"""

import numpy as np
from typing import Tuple
from models import ElasticProperties
from constants import PI, TWOPI, RAD_TO_DEG
from tensor_core import TensorOperations
from symmetry import SymmetryIdentifier
from averages import AverageCalculator
from anisotropy import AnisotropyCalculator
from directional import DirectionalProperties
from thermoacoustic import ThermoAcousticProperties

class ElasticTensor:
    """
    Classe para manipular o tensor de rigidez elástica Cij (6x6)
    e calcular propriedades elásticas seguindo a metodologia ElastPy.
    """
    
    def __init__(self, C: np.ndarray, density: float = 1.0, 
                 material_name: str = "Unknown"):
        """
        Inicializa com a matriz de rigidez.
        
        Args:
            C: Matriz de rigidez 6x6 em GPa
            density: Densidade do material em kg/m³
            material_name: Nome do material
        """
        self.C = np.array(C, dtype=float)
        self.density = density
        self.name = material_name
        self.tensor_ops = TensorOperations()
        
        # Identificar simetria
        self.symmetry = SymmetryIdentifier.identify(self.C)
        
        # Verificar dimensões
        if self.C.shape != (6, 6):
            raise ValueError("Matriz C deve ser 6x6")
        
        # Calcular matriz de compliância S
        self.S = self._calc_compliance()
        
        # Converter para tensor completo de 4ª ordem
        self.S4 = self.tensor_ops.voigt_to_tensor(self.S)
        
        # Verificar estabilidade
        self.is_stable, self.stability_message = self._check_stability()
        
        # Propriedades calculadas
        self.properties = ElasticProperties()
        self.properties.eigenvalues = self._calc_eigenvalues()
        
        # Inicializar calculadoras
        self.avg_calc = AverageCalculator()
        self.aniso_calc = AnisotropyCalculator()
        self.dir_props = DirectionalProperties(self.C, self.S, self.S4)
        self.thermo_acoustic = ThermoAcousticProperties()
        
        # Calcular todas as propriedades
        self._calculate_all_properties()
    
    def _calc_compliance(self) -> np.ndarray:
        """Calcula a matriz de compliância S = C^(-1)"""
        return np.linalg.inv(self.C)
    
    def _calc_eigenvalues(self) -> np.ndarray:
        """Calcula os autovalores da matriz de rigidez"""
        return np.linalg.eigvalsh(self.C)
    
    def _check_stability(self) -> Tuple[bool, str]:
        """Verifica condições de estabilidade elástica"""
        eigenvals = self._calc_eigenvalues()
        all_positive = np.all(eigenvals > 0)
        
        C = self.C
        minors = [
            C[0,0],
            C[0,0]*C[1,1] - C[0,1]**2,
            np.linalg.det(C[:3,:3])
        ]
        
        all_minors_positive = all(m > 0 for m in minors)
        
        if all_positive and all_minors_positive:
            return True, "The material is elastically stable."
        elif all_positive:
            return True, "The material is stable, although some minor criteria are not satisfied."
        else:
            return False, "The material exhibits elastic instability."
    
    def _calculate_all_properties(self):
        """Calcula todas as propriedades elásticas"""
        # Propriedades médias
        self.properties.B_V, self.properties.G_V = self.avg_calc.voigt(self.C)
        self.properties.B_R, self.properties.G_R = self.avg_calc.reuss(self.S)
        self.properties.B_H, self.properties.G_H, self.properties.E_H, self.properties.nu_H = \
            self.avg_calc.hill(self.properties.B_V, self.properties.G_V,
                               self.properties.B_R, self.properties.G_R)
        
        # Voigt e Reuss para E e ν
        self.properties.E_V = 9 * self.properties.B_V * self.properties.G_V / \
                              (3 * self.properties.B_V + self.properties.G_V)
        self.properties.E_R = 9 * self.properties.B_R * self.properties.G_R / \
                              (3 * self.properties.B_R + self.properties.G_R)
        self.properties.nu_V = (3 * self.properties.B_V - 2 * self.properties.G_V) / \
                                (2 * (3 * self.properties.B_V + self.properties.G_V))
        self.properties.nu_R = (3 * self.properties.B_R - 2 * self.properties.G_R) / \
                                (2 * (3 * self.properties.B_R + self.properties.G_R))
        
        # Razões e anisotropia
        self.properties.Pugh_ratio = self.properties.B_H / self.properties.G_H
        self.properties.A_U, self.properties.A_comp = self.aniso_calc.universal(
            self.properties.B_V, self.properties.G_V,
            self.properties.B_R, self.properties.G_R
        )
        self.properties.A_shear = self.aniso_calc.shear(self.C)
        
        # Valores extremos direcionais
        self.max_min_properties_elate_improved()
        self.properties.A_young = self.aniso_calc.young(
            self.properties.E_max, self.properties.E_min
        )
        
        # Propriedades acústicas
        vp, vs, vm, ratio = self.thermo_acoustic.sound_velocities(
            self.properties.B_H, self.properties.G_H, self.density
        )
        self.properties.vp = vp
        self.properties.vs = vs
        self.properties.vm = vm
        self.properties.vp_vs_ratio = ratio
        
        # Índices de anisotropia adicionais
        self.properties.Zener_anisotropy = self.aniso_calc.zener(self.C, self.symmetry)
        self.properties.Chung_anisotropy = self.aniso_calc.chung(
            self.properties.B_V, self.properties.G_V,
            self.properties.B_R, self.properties.G_R
        )
        
        # Dureza estimada
        self.properties.hardness = self.thermo_acoustic.hardness(
            self.properties.B_H, self.properties.G_H
        )
    
    def direction_vector(self, theta: float, phi: float) -> np.ndarray:
        """Vetor direção a partir de ângulos esféricos"""
        return self.tensor_ops.direction_vector(theta, phi)
    
    def angles_from_vector(self, n: np.ndarray) -> Tuple[float, float]:
        """Ângulos esféricos a partir de vetor direção"""
        return self.tensor_ops.angles_from_vector(n)
    
    def young_modulus_directional(self, n: np.ndarray) -> float:
        """Módulo de Young na direção n"""
        return self.dir_props.young_modulus(n)
    
    def linear_compressibility(self, n: np.ndarray) -> float:
        """Compressibilidade linear na direção n"""
        return self.dir_props.linear_compressibility(n)
    
    def shear_modulus_directional_improved(self, n: np.ndarray, nsteps_chi: int = 180) -> Tuple[float, float]:
        """Módulo de cisalhamento com χ ∈ [0, 2π]"""
        return self.dir_props.shear_modulus_improved(n, nsteps_chi)
    
    def poisson_ratio_directional_improved(self, n: np.ndarray, nsteps_chi: int = 180) -> Tuple[float, float]:
        """Coeficiente de Poisson com χ ∈ [0, 2π]"""
        return self.dir_props.poisson_ratio_improved(n, nsteps_chi)
    
    def max_min_properties_elate_improved(self, nsteps_theta: int = 30, 
                                          nsteps_phi: int = 60,
                                          nsteps_chi: int = 180) -> None:
        """
        Calcula valores máximos e mínimos das propriedades direcionais
        seguindo a metodologia ElastPy com melhorias de precisão.
        """
        theta_vals = np.linspace(0, PI, nsteps_theta, endpoint=False)
        phi_vals = np.linspace(0, TWOPI, nsteps_phi, endpoint=False)
        
        # Inicializar com valores extremos
        E_max, E_min = -np.inf, np.inf
        G_global_min, G_global_max = np.inf, -np.inf
        beta_max, beta_min = -np.inf, np.inf
        nu_global_min, nu_global_max = np.inf, -np.inf
        
        E_max_dir = E_min_dir = None
        G_min_dir = G_max_dir = None
        beta_max_dir = beta_min_dir = None
        nu_min_dir = nu_max_dir = None
        
        total = len(theta_vals) * len(phi_vals)
        count = 0
        
        print("Calculating directional properties ...")
        print(f"  Grid: {nsteps_theta}×{nsteps_phi} = {total} directions")
        print(f"  χ steps: {nsteps_chi} (full interval [0,2π])")
        
        for theta in theta_vals:
            for phi in phi_vals:
                count += 1
                if count % 500 == 0:
                    print(f"  In progress: {count}/{total} ({count/total*100:.1f}%)")
                
                n = self.direction_vector(theta, phi)
                
                # Módulo de Young
                E = self.young_modulus_directional(n)
                if E > E_max:
                    E_max = E
                    E_max_dir = n.copy()
                    self.properties.E_max_angles = (theta*RAD_TO_DEG, phi*RAD_TO_DEG)
                if E < E_min:
                    E_min = E
                    E_min_dir = n.copy()
                    self.properties.E_min_angles = (theta*RAD_TO_DEG, phi*RAD_TO_DEG)
                
                # Compressibilidade linear
                beta = self.linear_compressibility(n)
                if beta > beta_max:
                    beta_max = beta
                    beta_max_dir = n.copy()
                    self.properties.beta_max_angles = (theta*RAD_TO_DEG, phi*RAD_TO_DEG)
                if beta < beta_min:
                    beta_min = beta
                    beta_min_dir = n.copy()
                    self.properties.beta_min_angles = (theta*RAD_TO_DEG, phi*RAD_TO_DEG)
                
                # Shear modulus
                G_min_local, G_max_local = self.shear_modulus_directional_improved(n, nsteps_chi)
                
                if G_min_local < G_global_min:
                    G_global_min = G_min_local
                    G_min_dir = n.copy()
                    self.properties.G_min_angles = (theta*RAD_TO_DEG, phi*RAD_TO_DEG)
                
                if G_max_local > G_global_max:
                    G_global_max = G_max_local
                    G_max_dir = n.copy()
                    self.properties.G_max_angles = (theta*RAD_TO_DEG, phi*RAD_TO_DEG)
                
                # Poisson's ratio
                nu_min_local, nu_max_local = self.poisson_ratio_directional_improved(n, nsteps_chi)
                
                if nu_min_local < nu_global_min:
                    nu_global_min = nu_min_local
                    nu_min_dir = n.copy()
                    self.properties.nu_min_angles = (theta*RAD_TO_DEG, phi*RAD_TO_DEG)
                
                if nu_max_local > nu_global_max:
                    nu_global_max = nu_max_local
                    nu_max_dir = n.copy()
                    self.properties.nu_max_angles = (theta*RAD_TO_DEG, phi*RAD_TO_DEG)
        
        # Armazenar resultados
        self.properties.E_max = E_max
        self.properties.E_min = E_min
        self.properties.E_max_dir = E_max_dir
        self.properties.E_min_dir = E_min_dir
        self.properties.E_anisotropy = E_max / E_min if E_min > 0 else np.inf
        
        self.properties.G_max = G_global_max
        self.properties.G_min = G_global_min
        self.properties.G_max_dir = G_max_dir
        self.properties.G_min_dir = G_min_dir
        self.properties.G_anisotropy = G_global_max / G_global_min if G_global_min > 0 else np.inf
        
        self.properties.beta_max = beta_max
        self.properties.beta_min = beta_min
        self.properties.beta_max_dir = beta_max_dir
        self.properties.beta_min_dir = beta_min_dir
        self.properties.beta_anisotropy = beta_max / beta_min if beta_min > -100 else np.inf
                
        self.properties.nu_max = nu_global_max
        self.properties.nu_min = nu_global_min
        self.properties.nu_max_dir = nu_max_dir
        self.properties.nu_min_dir = nu_min_dir
        self.properties.nu_anisotropy = nu_global_max / nu_global_min if abs(nu_global_min) > 1e-6 else np.inf
        
        print(f"  Calculation successfully completed!")
    
    def generate_spherical_data_improved(self, nsteps: int = 30) -> dict:
        """Gera dados das propriedades em coordenadas esféricas"""
        from tensor_core import TensorOperations
        
        theta = np.linspace(0, PI, nsteps)
        phi = np.linspace(0, TWOPI, 2*nsteps)
        Theta, Phi = np.meshgrid(theta, phi)
        
        E_data = np.zeros_like(Theta)
        beta_data = np.zeros_like(Theta)
        G_min_data = np.zeros_like(Theta)
        G_max_data = np.zeros_like(Theta)
        nu_min_data = np.zeros_like(Theta)
        nu_max_data = np.zeros_like(Theta)
        
        for i in range(len(phi)):
            for j in range(len(theta)):
                n = TensorOperations.direction_vector(Theta[i,j], Phi[i,j])
                
                E_data[i,j] = self.young_modulus_directional(n)
                beta_data[i,j] = self.linear_compressibility(n)
                
                G_min, G_max = self.shear_modulus_directional_improved(n)
                G_min_data[i,j] = G_min
                G_max_data[i,j] = G_max
                
                nu_min, nu_max = self.poisson_ratio_directional_improved(n)
                nu_min_data[i,j] = nu_min
                nu_max_data[i,j] = nu_max
        
        return {
            'theta': Theta,
            'phi': Phi,
            'young': E_data,
            'compressibility': beta_data,
            'shear_min': G_min_data,
            'shear_max': G_max_data,
            'poisson_min': nu_min_data,
            'poisson_max': nu_max_data
        }
    
    def generate_polar_data_improved(self, plane: str = 'xy', npoints: int = 100) -> dict:
        """Gera dados para plotagem polar em um plano específico"""
        from constants import TWOPI
        from tensor_core import TensorOperations
        
        angles = np.linspace(0, TWOPI, npoints)
        
        E_data = np.zeros(npoints)
        beta_data = np.zeros(npoints)
        G_min_data = np.zeros(npoints)
        G_max_data = np.zeros(npoints)
        nu_min_data = np.zeros(npoints)
        nu_max_data = np.zeros(npoints)
        
        for i, alpha in enumerate(angles):
            if plane == 'xy':
                n = np.array([np.cos(alpha), np.sin(alpha), 0])
            elif plane == 'xz':
                n = np.array([np.cos(alpha), 0, np.sin(alpha)])
            elif plane == 'yz':
                n = np.array([0, np.cos(alpha), np.sin(alpha)])
            else:
                raise ValueError(f"Plane '{plane}' not recognized.")
            
            E_data[i] = self.young_modulus_directional(n)
            beta_data[i] = self.linear_compressibility(n)
            
            G_min, G_max = self.shear_modulus_directional_improved(n)
            G_min_data[i] = G_min
            G_max_data[i] = G_max
            
            nu_min, nu_max = self.poisson_ratio_directional_improved(n)
            nu_min_data[i] = nu_min
            nu_max_data[i] = nu_max
        
        return {
            'angles': angles,
            'young': E_data,
            'compressibility': beta_data,
            'shear_min': G_min_data,
            'shear_max': G_max_data,
            'poisson_min': nu_min_data,
            'poisson_max': nu_max_data
        }

"""
Utility functions
"""

import numpy as np
from datetime import datetime
from core import ElasticTensor

def read_cij_file(filename: str) -> np.ndarray:
    """Lê arquivo com matriz de rigidez Cij"""
    try:
        data = np.loadtxt(filename)
        if data.shape == (6, 6):
            return data
        else:
            data = data.flatten()
            if len(data) == 36:
                return data.reshape(6, 6)
            else:
                raise ValueError(f"Invalid format: 36 values expected, but found {len(data)}")
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return None

def save_report_improved(et: ElasticTensor, filename: str = "ElastPy_report.txt"):
    """Salva relatório em arquivo texto"""
    with open(filename, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"ElastPy - ELASTIC PROPERTIES ANALYSIS REPORT\n")
        f.write(f"Material: {et.name}\n")
        f.write(f"Symmetry: {et.symmetry}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("STIFFNESS MATRIX Cij (GPa):\n")
        f.write("-" * 50 + "\n")
        for i in range(6):
            f.write("  ".join([f"{et.C[i,j]:10.2f}" for j in range(6)]) + "\n")
        f.write("\n")
        
        f.write(f"Stability: {et.stability_message}\n\n")
        
        f.write("EIGENVALUES (GPa):\n")
        for i, val in enumerate(et.properties.eigenvalues, 1):
            f.write(f"  λ{i} = {val:10.2f}\n")
        f.write("\n")
        
        f.write("HILL AVERAGES:\n")
        f.write(f"  Bulk Modulus B      = {et.properties.B_H:10.2f} GPa\n")
        f.write(f"  Shear Modulus G     = {et.properties.G_H:10.2f} GPa\n")
        f.write(f"  Young's Modulus E   = {et.properties.E_H:10.2f} GPa\n")
        f.write(f"  Poisson's Ratio ν   = {et.properties.nu_H:10.4f}\n")
        f.write(f"  Pugh's Ratio B/G    = {et.properties.Pugh_ratio:10.4f}\n\n")
        
        f.write("ANISOTROPY INDICES (ElastPy ):\n")
        f.write(f"  Universal A_U       = {et.properties.A_U:10.4f}\n")
        f.write(f"  Compressibility     = {et.properties.A_comp:10.2f}%\n")
        f.write(f"  Zener Anisotropy    = {et.properties.Zener_anisotropy:10.4f}\n")
        f.write(f"  Chung Anisotropy    = {et.properties.Chung_anisotropy:10.4f}\n\n")
        
        f.write("EXTREME VALUES (ElastPy  METHOD):\n")
        f.write("-" * 60 + "\n")
        f.write(f"Young's Modulus:\n")
        f.write(f"  Max = {et.properties.E_max:8.2f} GPa at θ={et.properties.E_max_angles[0]:.1f}°, φ={et.properties.E_max_angles[1]:.1f}°\n")
        f.write(f"  Min = {et.properties.E_min:8.2f} GPa at θ={et.properties.E_min_angles[0]:.1f}°, φ={et.properties.E_min_angles[1]:.1f}°\n")
        f.write(f"  Anisotropy = {et.properties.E_anisotropy:.4f}\n\n")
        
        f.write(f"Shear Modulus (over χ ∈ [0, 2π]):\n")
        f.write(f"  Max = {et.properties.G_max:8.2f} GPa at θ={et.properties.G_max_angles[0]:.1f}°, φ={et.properties.G_max_angles[1]:.1f}°\n")
        f.write(f"  Min = {et.properties.G_min:8.2f} GPa at θ={et.properties.G_min_angles[0]:.1f}°, φ={et.properties.G_min_angles[1]:.1f}°\n")
        f.write(f"  Anisotropy = {et.properties.G_anisotropy:.4f}\n\n")
        
        f.write(f"Linear Compressibility:\n")
        f.write(f"  Max = {et.properties.beta_max:8.4f} TPa⁻¹ at θ={et.properties.beta_max_angles[0]:.1f}°, φ={et.properties.beta_max_angles[1]:.1f}°\n")
        f.write(f"  Min = {et.properties.beta_min:8.4f} TPa⁻¹ at θ={et.properties.beta_min_angles[0]:.1f}°, φ={et.properties.beta_min_angles[1]:.1f}°\n")
        f.write(f"  Anisotropy = {et.properties.beta_anisotropy:.4f}\n\n")
        
        f.write(f"Poisson's Ratio (over χ ∈ [0, 2π]):\n")
        f.write(f"  Max = {et.properties.nu_max:8.4f} at θ={et.properties.nu_max_angles[0]:.1f}°, φ={et.properties.nu_max_angles[1]:.1f}°\n")
        f.write(f"  Min = {et.properties.nu_min:8.4f} at θ={et.properties.nu_min_angles[0]:.1f}°, φ={et.properties.nu_min_angles[1]:.1f}°\n")
        f.write(f"  Anisotropy = {et.properties.nu_anisotropy:.4f}\n\n")
        
        f.write("SOUND VELOCITIES:\n")
        f.write(f"  vp (longitudinal) = {et.properties.vp:8.2f} km/s\n")
        f.write(f"  vs (shear)        = {et.properties.vs:8.2f} km/s\n")
        f.write(f"  vm (mean)         = {et.properties.vm:8.2f} km/s\n")
        f.write(f"  vp/vs ratio       = {et.properties.vp_vs_ratio:8.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"ElastPy report saved to: {filename}")

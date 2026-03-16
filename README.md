ElastPy - Elasticity Analyzer Tool

https://img.shields.io/badge/python-3.7%252B-blue

https://img.shields.io/badge/license-MIT-green

https://img.shields.io/badge/version-1.0.26-orange

ElastPy is a comprehensive tool for analyzing anisotropic elastic properties of 3D materials, based on the methodology of Gaillac et al. (J. Phys.: Condens. Matter 28 (2016) 275201). The program calculates and visualizes elastic properties from the material's stiffness matrix Cij (6x6).

✨ Features
    - Precise calculation of Poisson's ratio with full χ ∈ [0, 2π] sweep 
    - Adaptive extremum search using numerical optimization 
    - Numerically stable orthogonal basis via QR decomposition 
    - Vectorized operations with np.einsum for computational efficiency 
    - Crystal symmetry exploitation (cubic, hexagonal, tetragonal, orthorhombic, etc.) 
    - Interactive 3D visualizations of directional properties 
    - Comprehensive text reports

📊 Calculated Properties

1. Averages (Voigt-Reuss-Hill)
    - Bulk modulus (B)
    - Shear modulus (G)
    - Young's modulus (E)
    - Poisson's ratio (ν)
2. Anisotropy Indices
    - Universal index Aᵤ
    - Compressibility anisotropy
    - Zener anisotropy
    - Chung anisotropy
3. Directional Extreme Values
    - Maximum and minimum Young's modulus
    - Maximum and minimum shear modulus (over χ)
    - Maximum and minimum Poisson's ratio (over χ)
    - Linear compressibility (positive/negative)
4. Acoustic Properties
    - Longitudinal wave velocity (vₚ)
    - Shear wave velocity (vₛ)
    - Mean wave velocity (vₘ)
    - vₚ/vₛ ratio

🚀 Installation

Requirements<br>
    - Python 3.7 or higher
    - NumPy
    - SciPy
    - Matplotlib

# Clone the repository
git clone https://github.com/your-username/elastpy.git

cd elastpy

# Install dependencies
pip install numpy scipy matplotlib

# Run
python3 __main__.py

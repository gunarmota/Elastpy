"""
Elastic properties visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection
from typing import Optional
import tkinter

from constants import RAD_TO_DEG, TWOPI, PI
from core import ElasticTensor

class ElasticPlotterELATE:
    """Classe para visualização das propriedades elásticas seguindo ElastPy"""
    
    def __init__(self, etensor: ElasticTensor):
        self.et = etensor
        self.props = etensor.properties
        plt.ioff()
    
    def _safe_show(self, block=True):
        """Método seguro para exibir figuras evitando erros Tkinter"""
        try:
            if plt.get_fignums():
                if block:
                    plt.show(block=True)
                else:
                    plt.show(block=False)
                    plt.pause(0.1)
        except (tkinter.TclError, AttributeError, RuntimeError):
            pass
        except Exception as e:
            print(f"Warning: Could not display figure: {e}")
    
    def plot_stiffness_matrix(self, save_path: str = None):
        """Plota matriz de rigidez"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        C = self.et.C
        im = ax.imshow(C, cmap='viridis', aspect='auto', vmin=0, vmax=np.max(C))
        
        for i in range(6):
            for j in range(6):
                color = 'white' if C[i,j] > np.max(C)/2 else 'black'
                ax.text(j, i, f'{C[i,j]:.1f}',
                       ha="center", va="center", color=color, fontsize=10)
        
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_xticklabels(['C11', 'C22', 'C33', 'C44', 'C55', 'C66'])
        ax.set_yticklabels(['C11', 'C22', 'C33', 'C44', 'C55', 'C66'])
        ax.set_title(f'Stiffness Matrix Cij (GPa) - {self.et.name}\nSymmetry: {self.et.symmetry}', 
                    fontsize=14, fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='GPa')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self._safe_show(block=True)
    
    def plot_young_3d(self, save_path: str = None, elev: int = 30, azim: int = 45):
        """Plota Young's modulus 3D"""
        data = self.et.generate_spherical_data_improved(nsteps=40)
        
        Theta = data['theta']
        Phi = data['phi']
        values = data['young']
        
        X = values * np.sin(Theta) * np.cos(Phi)
        Y = values * np.sin(Theta) * np.sin(Phi)
        Z = values * np.cos(Theta)
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        norm = plt.Normalize(self.props.E_min, self.props.E_max)
        surf = ax.plot_surface(X, Y, Z, facecolors=plt.cm.plasma(norm(values)),
                              alpha=0.95, linewidth=0, antialiased=True)
        
        max_val = np.max([np.abs(X), np.abs(Y), np.abs(Z)])
        ax.plot([-max_val, max_val], [0, 0], [0, 0], 'k-', alpha=0.3, linewidth=0.5)
        ax.plot([0, 0], [-max_val, max_val], [0, 0], 'k-', alpha=0.3, linewidth=0.5)
        ax.plot([0, 0], [0, 0], [-max_val, max_val], 'k-', alpha=0.3, linewidth=0.5)
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title(f"Young's Modulus (GPa) - {self.et.name}", fontsize=14, fontweight='bold')
        
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
        ax.set_zlim(-max_val, max_val)
        ax.view_init(elev=elev, azim=azim)
        
        mappable = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
        mappable.set_array(values)
        cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=20, pad=0.1, alpha=0.7)
        cbar.set_label('GPa', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self._safe_show(block=True)
    
    def plot_compressibility_3d(self, save_path: str = None, elev: int = 30, azim: int = 45):
        """Plota linear compressibility 3D"""
        data = self.et.generate_spherical_data_improved(nsteps=40)
        
        Theta = data['theta']
        Phi = data['phi']
        values = data['compressibility']
        
        X = values * np.sin(Theta) * np.cos(Phi)
        Y = values * np.sin(Theta) * np.sin(Phi)
        Z = values * np.cos(Theta)
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        mask_pos = values >= 0
        mask_neg = values < 0
        
        if np.any(mask_pos):
            ax.plot_surface(X * mask_pos, Y * mask_pos, Z * mask_pos,
                          color='green', alpha=0.8, linewidth=0)
        
        if np.any(mask_neg):
            ax.plot_surface(X * mask_neg, Y * mask_neg, Z * mask_neg,
                          color='red', alpha=0.8, linewidth=0)
        
        max_val = np.max([np.abs(X), np.abs(Y), np.abs(Z)])
        ax.plot([-max_val, max_val], [0, 0], [0, 0], 'k-', alpha=0.3, linewidth=0.5)
        ax.plot([0, 0], [-max_val, max_val], [0, 0], 'k-', alpha=0.3, linewidth=0.5)
        ax.plot([0, 0], [0, 0], [-max_val, max_val], 'k-', alpha=0.3, linewidth=0.5)
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title(f"Linear Compressibility (1/TPa) - {self.et.name}", fontsize=14, fontweight='bold')
        
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
        ax.set_zlim(-max_val, max_val)
        ax.view_init(elev=elev, azim=azim)
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.8, label='Positive'),
                          Patch(facecolor='red', alpha=0.8, label='Negative')]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self._safe_show(block=True)
    
    def plot_shear_3d(self, save_path: str = None, elev: int = 30, azim: int = 45):
        """Plota shear modulus 3D"""
        data = self.et.generate_spherical_data_improved(nsteps=40)
        
        Theta = data['theta']
        Phi = data['phi']
        G_min = data['shear_min']
        G_max = data['shear_max']
        
        X_min = G_min * np.sin(Theta) * np.cos(Phi)
        Y_min = G_min * np.sin(Theta) * np.sin(Phi)
        Z_min = G_min * np.cos(Theta)
        
        X_max = G_max * np.sin(Theta) * np.cos(Phi)
        Y_max = G_max * np.sin(Theta) * np.sin(Phi)
        Z_max = G_max * np.cos(Theta)
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot_surface(X_max, Y_max, Z_max, color='blue', alpha=0.3, linewidth=0)
        ax.plot_surface(X_min, Y_min, Z_min, color='green', alpha=0.8, linewidth=0)
        
        max_val = np.max([np.abs(X_max), np.abs(Y_max), np.abs(Z_max)])
        ax.plot([-max_val, max_val], [0, 0], [0, 0], 'k-', alpha=0.3, linewidth=0.5)
        ax.plot([0, 0], [-max_val, max_val], [0, 0], 'k-', alpha=0.3, linewidth=0.5)
        ax.plot([0, 0], [0, 0], [-max_val, max_val], 'k-', alpha=0.3, linewidth=0.5)
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title(f"Shear Modulus (GPa) - {self.et.name}", fontsize=14, fontweight='bold')
        
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
        ax.set_zlim(-max_val, max_val)
        ax.view_init(elev=elev, azim=azim)
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.8, label=f'Min G = {self.props.G_min:.1f} GPa'),
                          Patch(facecolor='blue', alpha=0.3, label=f'Max G = {self.props.G_max:.1f} GPa')]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self._safe_show(block=True)
    
    def plot_poisson_3d(self, save_path: str = None, elev: int = 30, azim: int = 45):
        """Plota Poisson's ratio 3D"""
        data = self.et.generate_spherical_data_improved(nsteps=40)
        
        Theta = data['theta']
        Phi = data['phi']
        nu_min = data['poisson_min']
        nu_max = data['poisson_max']
        
        X_min = nu_min * np.sin(Theta) * np.cos(Phi)
        Y_min = nu_min * np.sin(Theta) * np.sin(Phi)
        Z_min = nu_min * np.cos(Theta)
        
        X_max = nu_max * np.sin(Theta) * np.cos(Phi)
        Y_max = nu_max * np.sin(Theta) * np.sin(Phi)
        Z_max = nu_max * np.cos(Theta)
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot_surface(X_max, Y_max, Z_max, color='blue', alpha=0.2, linewidth=0)
        
        mask_pos = nu_min >= 0
        mask_neg = nu_min < 0
        
        if np.any(mask_pos):
            ax.plot_surface(X_min * mask_pos, Y_min * mask_pos, Z_min * mask_pos,
                          color='green', alpha=0.8, linewidth=0)
        
        if np.any(mask_neg):
            ax.plot_surface(X_min * mask_neg, Y_min * mask_neg, Z_min * mask_neg,
                          color='red', alpha=0.8, linewidth=0)
        
        max_val = np.max([np.abs(X_max), np.abs(Y_max), np.abs(Z_max)])
        ax.plot([-max_val, max_val], [0, 0], [0, 0], 'k-', alpha=0.3, linewidth=0.5)
        ax.plot([0, 0], [-max_val, max_val], [0, 0], 'k-', alpha=0.3, linewidth=0.5)
        ax.plot([0, 0], [0, 0], [-max_val, max_val], 'k-', alpha=0.3, linewidth=0.5)
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title(f"Poisson's Ratio - {self.et.name}", fontsize=14, fontweight='bold')
        
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
        ax.set_zlim(-max_val, max_val)
        ax.view_init(elev=elev, azim=azim)
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.8, label=f'Min ν (positive)'),
                          Patch(facecolor='red', alpha=0.8, label=f'Min ν (negative)'),
                          Patch(facecolor='blue', alpha=0.2, label='Max ν')]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self._safe_show(block=True)
    
    def plot_polar_projections_improved(self, save_path: str = None):
        """Plota projeções polares nos três planos principais"""
        planes = ['xy', 'xz', 'yz']
        plane_names = ['XY Plane (z=0)', 'XZ Plane (y=0)', 'YZ Plane (x=0)']
        
        fig, axes = plt.subplots(4, 3, figsize=(15, 16), subplot_kw={'projection': 'polar'})
        
        for i, (plane, plane_name) in enumerate(zip(planes, plane_names)):
            data = self.et.generate_polar_data_improved(plane=plane, npoints=200)
            angles = data['angles']
            
            # Young's modulus
            ax = axes[0, i]
            ax.plot(angles, data['young'], 'b-', linewidth=2)
            ax.fill(angles, data['young'], alpha=0.3)
            ax.set_title(f"Young's Modulus\n{plane_name}", fontsize=10)
            ax.grid(True, alpha=0.5)
            
            # Compressibility
            ax = axes[1, i]
            ax.plot(angles, data['compressibility'], 'r-', linewidth=2)
            ax.fill(angles, data['compressibility'], alpha=0.3, color='red')
            ax.set_title(f"Linear Compressibility\n{plane_name}", fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Shear modulus
            ax = axes[2, i]
            ax.plot(angles, data['shear_max'], 'b--', linewidth=1.5, label='Max', alpha=0.7)
            ax.plot(angles, data['shear_min'], 'g-', linewidth=2, label='Min')
            ax.fill_between(angles, data['shear_min'], data['shear_max'], alpha=0.2, color='blue')
            ax.set_title(f"Shear Modulus\n{plane_name}", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.05, 1.0))
            
            # Poisson's ratio
            ax = axes[3, i]
            ax.plot(angles, data['poisson_max'], 'b--', linewidth=1.5, label='Max', alpha=0.7)
            ax.plot(angles, data['poisson_min'], 'g-', linewidth=2, label='Min')
            ax.fill_between(angles, data['poisson_min'], data['poisson_max'], alpha=0.2, color='blue')
            ax.plot(angles, np.zeros_like(angles), 'k--', alpha=0.3, linewidth=0.5)
            ax.set_title(f"Poisson's Ratio\n{plane_name}", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.05, 1.0))
        
        plt.suptitle(f'Polar Projections (ElastPy) - {self.et.name}', 
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self._safe_show(block=True)
    
    def plot_polar_projections_elate(self, save_path: str = None):
        """Plota projeções polares nos três planos principais seguindo ELATE"""
        planes = ['xy', 'xz', 'yz']
        plane_names = ['XY Plane (z=0)', 'XZ Plane (y=0)', 'YZ Plane (x=0)']
        
        fig, axes = plt.subplots(4, 3, figsize=(15, 16))
        
        for i, (plane, plane_name) in enumerate(zip(planes, plane_names)):
            data = self.et.generate_polar_data_improved(plane=plane, npoints=200)
            angles = data['angles'] * RAD_TO_DEG
            
            # Young's modulus
            ax = axes[0, i]
            ax.plot(angles, data['young'], 'b-', linewidth=2)
            ax.fill_between(angles, 0, data['young'], alpha=0.3)
            ax.set_xlabel('Angle (degrees)')
            ax.set_ylabel('GPa')
            ax.set_title(f"Young's Modulus\n{plane_name}")
            ax.grid(True, alpha=0.3)
            
            # Compressibility
            ax = axes[1, i]
            ax.plot(angles, data['compressibility'], 'r-', linewidth=2)
            ax.fill_between(angles, 0, data['compressibility'], alpha=0.3, color='red')
            ax.set_xlabel('Angle (degrees)')
            ax.set_ylabel('TPa⁻¹')
            ax.set_title(f"Linear Compressibility\n{plane_name}")
            ax.grid(True, alpha=0.3)
            
            # Shear modulus (min e max)
            ax = axes[2, i]
            ax.plot(angles, data['shear_max'], 'b--', linewidth=1.5, label='Max', alpha=0.7)
            ax.plot(angles, data['shear_min'], 'g-', linewidth=2, label='Min')
            ax.fill_between(angles, data['shear_min'], data['shear_max'], alpha=0.2, color='blue')
            ax.set_xlabel('Angle (degrees)')
            ax.set_ylabel('GPa')
            ax.set_title(f"Shear Modulus\n{plane_name}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            # Poisson's ratio (min e max)
            ax = axes[3, i]
            ax.plot(angles, data['poisson_max'], 'b--', linewidth=1.5, label='Max', alpha=0.7)
            ax.plot(angles, data['poisson_min'], 'g-', linewidth=2, label='Min')
            ax.fill_between(angles, data['poisson_min'], data['poisson_max'], alpha=0.2, color='blue')
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.set_xlabel('Angle (degrees)')
            ax.set_ylabel('ν')
            ax.set_title(f"Poisson's Ratio\n{plane_name}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        plt.suptitle(f'Polar Projections - {self.et.name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self._safe_show(block=True)
        #plt.show()
    
    def plot_comprehensive_report_improved(self, save_path: str = None):
        """Plota um relatório abrangente com todas as informações"""
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Matriz de rigidez
        ax1 = fig.add_subplot(3, 4, 1)
        C = self.et.C
        im = ax1.imshow(C, cmap='viridis', aspect='auto', vmin=0, vmax=np.max(C))
        ax1.set_title('Stiffness Matrix Cij (GPa)', fontsize=10)
        ax1.set_xticks(range(6))
        ax1.set_yticks(range(6))
        ax1.set_xticklabels(['C11', 'C22', 'C33', 'C44', 'C55', 'C66'], rotation=45)
        ax1.set_yticklabels(['C11', 'C22', 'C33', 'C44', 'C55', 'C66'])
        plt.colorbar(im, ax=ax1, shrink=0.8)
        
        # 2. Propriedades médias
        ax2 = fig.add_subplot(3, 4, 2)
        ax2.axis('off')
        text = f"""
        {self.et.name}
        Symmetry: {self.et.symmetry}
        {self.et.stability_message}
        
        VOIGT AVERAGES:
        B_V = {self.props.B_V:.2f} GPa
        G_V = {self.props.G_V:.2f} GPa
        E_V = {self.props.E_V:.2f} GPa
        ν_V = {self.props.nu_V:.3f}
        
        REUSS AVERAGES:
        B_R = {self.props.B_R:.2f} GPa
        G_R = {self.props.G_R:.2f} GPa
        E_R = {self.props.E_R:.2f} GPa
        ν_R = {self.props.nu_R:.3f}
        
        HILL AVERAGES:
        B_H = {self.props.B_H:.2f} GPa
        G_H = {self.props.G_H:.2f} GPa
        E_H = {self.props.E_H:.2f} GPa
        ν_H = {self.props.nu_H:.3f}
        """
        ax2.text(0.05, 0.95, text, fontsize=8, verticalalignment='top',
                family='monospace', transform=ax2.transAxes)
        
        # 3. Autovalores
        ax3 = fig.add_subplot(3, 4, 3)
        eigenvals = self.props.eigenvalues
        bars = ax3.bar(range(1, 7), eigenvals, color='steelblue')
        ax3.set_title('Eigenvalues', fontsize=10)
        ax3.set_xlabel('Mode')
        ax3.set_ylabel('Eigenvalue (GPa)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Índices de anisotropia
        ax4 = fig.add_subplot(3, 4, 4)
        ax4.axis('off')
        text = f"""
        ANISOTROPY INDICES (ElastPy ):
        
        Universal A_U = {self.props.A_U:.3f}
        Compressibility = {self.props.A_comp:.2f}%
        Shear A_shear = {self.props.A_shear:.3f}
        Young A_young = {self.props.A_young:.3f}
        Zener A_Z = {self.props.Zener_anisotropy:.3f}
        Chung A_C = {self.props.Chung_anisotropy:.3f}
        
        Pugh's Ratio = {self.props.Pugh_ratio:.3f}
        Ductile: {'Yes' if self.props.Pugh_ratio > 1.75 else 'No'}
        Hardness = {self.props.hardness:.2f} GPa
        """
        ax4.text(0.05, 0.95, text, fontsize=8, verticalalignment='top',
                family='monospace', transform=ax4.transAxes)
        
        # 5-8. Informações de extremos
        ax5 = fig.add_subplot(3, 4, 5)
        ax5.axis('off')
        text = f"""
        EXTREME VALUES (ElastPy ):
        
        YOUNG'S MODULUS:
        Max = {self.props.E_max:.1f} GPa
        at θ={self.props.E_max_angles[0]:.1f}°, φ={self.props.E_max_angles[1]:.1f}°
        Min = {self.props.E_min:.1f} GPa
        at θ={self.props.E_min_angles[0]:.1f}°, φ={self.props.E_min_angles[1]:.1f}°
        Anisotropy = {self.props.E_anisotropy:.3f}
        
        SHEAR MODULUS:
        Max = {self.props.G_max:.1f} GPa
        at θ={self.props.G_max_angles[0]:.1f}°, φ={self.props.G_max_angles[1]:.1f}°
        Min = {self.props.G_min:.1f} GPa
        at θ={self.props.G_min_angles[0]:.1f}°, φ={self.props.G_min_angles[1]:.1f}°
        Anisotropy = {self.props.G_anisotropy:.3f}
        """
        ax5.text(0.05, 0.95, text, fontsize=8, verticalalignment='top',
                family='monospace', transform=ax5.transAxes)
        
        ax6 = fig.add_subplot(3, 4, 6)
        ax6.axis('off')
        text = f"""
        LINEAR COMPRESSIBILITY:
        Max = {self.props.beta_max:.4f} TPa⁻¹
        at θ={self.props.beta_max_angles[0]:.1f}°, φ={self.props.beta_max_angles[1]:.1f}°
        Min = {self.props.beta_min:.4f} TPa⁻¹
        at θ={self.props.beta_min_angles[0]:.1f}°, φ={self.props.beta_min_angles[1]:.1f}°
        Anisotropy = {self.props.beta_anisotropy:.3f}
        
        POISSON'S RATIO (χ ∈ [0, 2π]):
        Max = {self.props.nu_max:.4f}
        at θ={self.props.nu_max_angles[0]:.1f}°, φ={self.props.nu_max_angles[1]:.1f}°
        Min = {self.props.nu_min:.4f}
        at θ={self.props.nu_min_angles[0]:.1f}°, φ={self.props.nu_min_angles[1]:.1f}°
        Anisotropy = {self.props.nu_anisotropy:.3f}
        """
        ax6.text(0.05, 0.95, text, fontsize=8, verticalalignment='top',
                family='monospace', transform=ax6.transAxes)
        
        ax7 = fig.add_subplot(3, 4, 7)
        ax7.axis('off')
        text = f"""
        SOUND VELOCITIES:
        vp = {self.props.vp:.2f} km/s
        vs = {self.props.vs:.2f} km/s
        vm = {self.props.vm:.2f} km/s
        vp/vs = {self.props.vp_vs_ratio:.3f}
        
        Density = {self.et.density:.1f} kg/m³
        """
        ax7.text(0.05, 0.95, text, fontsize=8, verticalalignment='top',
                family='monospace', transform=ax7.transAxes)
        
        ax8 = fig.add_subplot(3, 4, 8, projection='3d')
        data = self.et.generate_spherical_data_improved(nsteps=20)
        X = data['young'] * np.sin(data['theta']) * np.cos(data['phi'])
        Y = data['young'] * np.sin(data['theta']) * np.sin(data['phi'])
        Z = data['young'] * np.cos(data['theta'])
        ax8.plot_surface(X, Y, Z, alpha=0.8, cmap='plasma', linewidth=0)
        ax8.set_title("Young's Modulus 3D", fontsize=10)
        ax8.set_xticks([])
        ax8.set_yticks([])
        ax8.set_zticks([])
        
        # 9-12. Gráficos polares simplificados
        data_xy = self.et.generate_polar_data_improved(plane='xy', npoints=200)
        angles = data_xy['angles'] * RAD_TO_DEG
        
        ax9 = fig.add_subplot(3, 4, 9)
        ax9.plot(angles, data_xy['young'], 'b-')
        ax9.set_title("Young's Modulus (XY)")
        ax9.set_xlabel('Angle (°)')
        ax9.grid(True, alpha=0.3)
        
        ax10 = fig.add_subplot(3, 4, 10)
        ax10.plot(angles, data_xy['compressibility'], 'r-')
        ax10.set_title("Compressibility (XY)")
        ax10.set_xlabel('Angle (°)')
        ax10.grid(True, alpha=0.3)
        
        ax11 = fig.add_subplot(3, 4, 11)
        ax11.plot(angles, data_xy['shear_max'], 'b--', label='Max', alpha=0.7)
        ax11.plot(angles, data_xy['shear_min'], 'g-', label='Min')
        ax11.set_title("Shear Modulus (XY)")
        ax11.set_xlabel('Angle (°)')
        ax11.grid(True, alpha=0.3)
        ax11.legend(fontsize=6)
        
        ax12 = fig.add_subplot(3, 4, 12)
        ax12.plot(angles, data_xy['poisson_max'], 'b--', label='Max', alpha=0.7)
        ax12.plot(angles, data_xy['poisson_min'], 'g-', label='Min')
        ax12.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax12.set_title("Poisson's Ratio (XY)")
        ax12.set_xlabel('Angle (°)')
        ax12.grid(True, alpha=0.3)
        ax12.legend(fontsize=6)
        
        plt.suptitle(f'ElastPy Comprehensive Report - {self.et.name}', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self._safe_show(block=True)
    
    def validate_isotropic_case(self):
        """Valida o cálculo para um material isotrópico teórico"""
        print("\n" + "="*60)
        print("VALIDATION: Isotropic Material")
        print("="*60)
        
        C_iso = np.array([
            [200, 100, 100, 0, 0, 0],
            [100, 200, 100, 0, 0, 0],
            [100, 100, 200, 0, 0, 0],
            [0, 0, 0, 50, 0, 0],
            [0, 0, 0, 0, 50, 0],
            [0, 0, 0, 0, 0, 50]
        ])
        
        from .core import ElasticTensor
        et_iso = ElasticTensor(C_iso, material_name="Isotropic Test")
        
        test_dirs = [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
            np.array([1, 1, 0]) / np.sqrt(2),
            np.array([1, 1, 1]) / np.sqrt(3),
            np.array([1, 2, 3]) / np.sqrt(14)
        ]
        
        print("\nPoisson’s Ratio Test (should be constant):")
        nu_values = []
        
        for i, n in enumerate(test_dirs):
            nu_min, nu_max = et_iso.poisson_ratio_directional_improved(n)
            nu_mean = (nu_min + nu_max) / 2
            nu_values.append(nu_mean)
            print(f"  Direction {i+1}: ν_min={nu_min:.6f}, ν_max={nu_max:.6f}, média={nu_mean:.6f}")
        
        nu_mean = np.mean(nu_values)
        nu_std = np.std(nu_values)
        
        print(f"\n  ν mean = {nu_mean:.6f}")
        print(f"  Standard deviation = {nu_std:.6e}")
        
        if nu_std < 1e-4:
            print("Isotropy test passed!")
            return True
        else:
            print("Isotropy test failed!")
            return False

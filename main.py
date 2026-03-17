#!/usr/bin/env python3
"""
Elasticity Analyzer Tool (ElastPy) – v1.0.26
Analyzing anisotropic elastic properties of 3D materials
"""


import os
import sys

# Adiciona o diretório atual ao path do Python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from core import ElasticTensor
from plotting import ElasticPlotterELATE
from utils import read_cij_file, save_report_improved

def main():
    print("\n" + "="*80)
    print(" ELASTICITY ANALYZER TOOL (ElastPy)  VERSION 1.0.26")
    #print(" Based on Gaillac et al., J. Phys.: Condens. Matter 28 (2016) 275201")
    print(" Improved version with full χ range [0, 2π] and adaptive extremum search")
    print("="*80 + "\n")
    
    example_file = "Cij.dat"
    
    if not os.path.exists(example_file):
        print(f"Generating example file: {example_file}")
        example_data = np.array([
            [246.73, 126.66, 104.6, 0, 0, 0],
            [126.66, 246.73, 104.6, 0, 0, 0],
            [104.6, 104.6, 241.26, 0, 0, 0],
            [0, 0, 0, 56.48, 0, 0],
            [0, 0, 0, 0, 56.48, 0],
            [0, 0, 0, 0, 0, 60.04]
        ])
        np.savetxt(example_file, example_data, fmt='%8.2f')
    
    while True:
        print("\n" + "-"*50)
        print("MAIN MENU:")
        print("1. Analyze Cij.dat file (example: hexagonal ZnO)")
        print("2. Specify input file")
        print("3. Run validation (isotropic material)")
        print("4. Exit")
        print("-"*50)
        
        choice = input("Choose an option: ").strip()
        
        if choice == '1':
            filename = "Cij.dat"
        elif choice == '2':
            filename = input("File name: ").strip()
            if not os.path.exists(filename):
                print(f"❌ File {filename} not found!")
                continue
        elif choice == '3':
            et_temp = ElasticTensor(np.eye(6))
            plotter_temp = ElasticPlotterELATE(et_temp)
            plotter_temp.validate_isotropic_case()
            continue
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid option!")
            continue
        
        print(f"\n Reading file: {filename}")
        C = read_cij_file(filename)
        
        if C is None:
            continue
        
        #print("\nTable of typical densities (kg/m³):")
        #print("  Light metals: 2700-4500")
        #print("  Steels: 7800-8000")
        #print("  Ceramics: 3800-6000")
        #print("  Semiconductors: 2330-5320")
        
        density_input = input("\nMaterial density (kg/m³) [Press Enter to skip]: ").strip()
        density = float(density_input) if density_input else 1.0
        
        name = input("Material name [Press Enter for 'Unknown']: ").strip()
        if not name:
            name = "Unknown"
        
        print("\n Calculating elastic properties...")
        et = ElasticTensor(C, density=density, material_name=name)
        plotter = ElasticPlotterELATE(et)
        
        while True:
            print("\n" + "-"*50)
            print("VIEW OPTIONS:")
            print("1. Stiffness matrix")
            print("2. Average properties")
            print("3. Eigenvalues")
            print("4. 3D Young's Modulus")
            print("5. 3D Linear Compressibility")
            print("6. 3D Shear Modulus")
            print("7. 3D Poisson's Ratio)")
            print("8. XYZ polar projections")
            print("9. 2D polar projections")
            print("10. Full ElastPy report")
            print("11. Save report to file")
            print("12. Return to main menu")
            print("-"*50)
            
            viz_choice = input("Choose an option: ").strip()
            
            if viz_choice == '1':
                plotter.plot_stiffness_matrix()
            
            elif viz_choice == '2':
                print("\n" + "="*50)
                print("Hill averages:")
                print("="*50)
                print(f"Bulk Modulus B     = {et.properties.B_H:10.2f} GPa")
                print(f"Shear Modulus G    = {et.properties.G_H:10.2f} GPa")
                print(f"Young's Modulus E  = {et.properties.E_H:10.2f} GPa")
                print(f"Poisson's Ratio ν  = {et.properties.nu_H:10.4f}")
                print(f"Pugh's Ratio B/G   = {et.properties.Pugh_ratio:10.4f}")
                print("="*50)
            
            elif viz_choice == '3':
                print("\nEigenvalues (GPa):")
                for i, val in enumerate(et.properties.eigenvalues, 1):
                    print(f"  λ{i} = {val:10.2f}")
            
            elif viz_choice == '4':
                plotter.plot_young_3d()
            
            elif viz_choice == '5':
                plotter.plot_compressibility_3d()
            
            elif viz_choice == '6':
                plotter.plot_shear_3d()
            
            elif viz_choice == '7':
                plotter.plot_poisson_3d()

            elif viz_choice == '8':
                plotter.plot_polar_projections_elate()
                            
            elif viz_choice == '9':
                plotter.plot_polar_projections_improved()
            
            elif viz_choice == '10':
                plotter.plot_comprehensive_report_improved()
            
            elif viz_choice == '11':
                save_report_improved(et)
            
            elif viz_choice == '12':
                break
            
            else:
                print("Invalid option!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Program interrupted by user.")
    except Exception as e:
        print(f"\n❌ Runtime error: {e}")
        import traceback
        traceback.print_exc()

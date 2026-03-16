"""
Global constants and configuration
"""

import numpy as np
import matplotlib.pyplot as plt

# Constantes matemáticas
PI = np.pi
DEG_TO_RAD = PI / 180.0
RAD_TO_DEG = 180.0 / PI
TWOPI = 2 * PI

# Configurações de plotagem
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = plt.cm.viridis(np.linspace(0, 1, 10))

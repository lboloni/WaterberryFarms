import numpy as np
import matplotlib.pyplot as plt

def VoI(uncertainty, v_max, u_min, u_max):
    """Value of information function"""
    if uncertainty <  u_min:
        return v_max
    if uncertainty > u_max:
        return 0
    return v_max * (u_max - uncertainty) / (u_max - u_min)

range = np.arange(0, 1.0, 0.01)
voi = [VoI(x, 100, 0.3, 0.6) for x in range]

fig, ax = plt.subplots(1)

ax.plot(range, voi)

plt.show()

# note: SciPy can show the PDF of the distribions
# scipy.stats.norm.pdf()
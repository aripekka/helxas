# helxas
Python data extraction routines for HelXAS X-ray absorption instrument 

### Dependencies
Module requires silx, numpy and matplotlib. Easiest way to install them is using pip:
```
pip install --user silx numpy matplolib
```

## Example of use:
```python
from __future__ import division, print_function
from helxas import HelXAS
import matplotlib.pyplot as plt

helxas = HelXAS('some_datafile.spec','/path/to/data/folder')

I0_ind = [DIRECT BEAM SCAN INDICES]
I0bg_ind = [DIRECT BEAM BACKGROUND SCAN INDICES]

I_ind = [TRANSMITTED BEAM SCAN INDICES]
Ibg_ind = [TRANSMITTED BEAM BACKGROUND SCAN INDICES]

helxas.set_analyser('Si',533)
helxas.theta_calibration = 0 #Adjust this to calibrate the absolute theta/energy scale

helxas.read_I0(I0_ind,I0bg_ind)
helxas.read_I('my_sample',I_ind,Ibg_ind)

helxas.set_background_fit_order(3,True)
energy, mux, mux_error = helxas.get_spectrum('my_sample')

plt.figure()
plt.errorbar(energy, mux, mux_error)
plt.show()
```

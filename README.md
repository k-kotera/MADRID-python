# MADRID-python
Unofficial python implementation of MADRID algorithm, introduced in "Matrix Profile XXX: MADRID: A Hyper - Anytime and Parameter - Free Algorithm to Find Time Series Anomalies of all Lengths", ICDM2023.

# Example
```python
import numpy as np
import matplotlib.pyplot as plt
from madrid import MADRID

TS = np.loadtxt("UCR_Anomaly_FullData/119_UCR_Anomaly_ECG1_10000_11800_12100.txt")
train_test_split = 10000

best_discord_loc,_,_,_,_ = MADRID(TS, train_test_split)

plt.figure(figsize=(10,2))
plt.grid()
plt.plot(TS,lw=2., color="k")
plt.axvspan(best_discord_loc[0], best_discord_loc[1], facecolor='r', alpha=0.1,label="anomaly")
plt.xlim(best_discord_loc[0]-300,best_discord_loc[1]+300)
```
![](https://github.com/user-attachments/assets/a3b966a2-cfb8-4f6b-9873-f4846537b397)

## Requirements
- numpy
- numba
- rocket-fft
  - Required for FFT and IFFT to work in numba.njit
 
## Links
- Official MATLAB Code: 
https://sites.google.com/view/madrid-icdm-23


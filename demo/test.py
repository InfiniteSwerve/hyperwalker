import numpy as np
import torch as th

# Assuming tt is some data (example as list)
tt = [1, 2, 3, 4, 5]  # Example data

# Check if tt is a NumPy array, and convert if necessary
if not isinstance(tt, np.ndarray):
    tt = np.array(tt)

# Now, convert tt to a PyTorch tensor
tensor_tt = th.from_numpy(tt)
print(tensor_tt)

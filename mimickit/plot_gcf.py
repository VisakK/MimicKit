import torch 
import matplotlib.pyplot as plt
import numpy as np

# load tensor from file 
gcf = torch.load("./mimickit/ground_contact_forces_log_run.pt").squeeze(1)

gcf = gcf.numpy()  # convert to numpy array for easier handling
print(f"Ground Contact Forces shape: {gcf.shape}")  # should be (num_steps, num_envs, num_contact_bodies, 3)
num_steps, num_contact_bodies, _ = gcf.shape

plt.plot(gcf[:,-1,2])
plt.show()
import pickle 

path = "./data/motions/humanoid/humanoid_walk.pkl"

#load the pickle file
with open(path, 'rb') as f:
    data = pickle.load(f)

import pdb; pdb.set_trace()
    
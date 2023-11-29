#%%
import slim4
import numpy as np

mesh = slim4.d2.Mesh("square.msh")

xy = mesh.x.T
neighbours = mesh.neighbours.T

with open("square.txt", "w") as f:
    f.write(f"{len(xy)}\n")
    # write in .12 precision
    for(coord, n) in zip(xy, neighbours):
        for node in range(3):
            y = coord[node,1]
            gamma = 1e-1
            cor =  1e-4 + 2e-11*y
            taux = 0.1*np.sin(np.pi*y/1e6)
            tauy = 0.0
            bath = 1000
            c = np.sqrt(9.81*bath)
            f.write(f"{coord[node,0]:20.6e},{coord[node,1]:20.6e},{bath:20.6e},{gamma:20.6e},{cor:20.6e},{taux:20.6e},{tauy:20.6e},{c:20.6e},{n[node,0]:10d},{n[node,1]:10d}\n")

# %%

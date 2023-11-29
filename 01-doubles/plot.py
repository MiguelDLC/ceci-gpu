import numpy as np
from matplotlib import pyplot as plt

with open("../data/square_gmsh.txt", "r") as f:
    line = f.readline()
    n = int(line)
    xy = np.zeros((n,3,2))
    
    for i in range(n):
        for node in range(3):
            line = f.readline()
            line = line.split(",")
            xy[i,node,0] = float(line[0])
            xy[i,node,1] = float(line[1])

with open("solution.txt", "r") as f:
    sol = np.zeros((n,3,3))    
    for i in range(n):
        for node in range(3):
            line = f.readline()
            line = line.split(",")
            sol[i,node,0] = float(line[0])
            sol[i,node,1] = float(line[1])
            sol[i,node,2] = float(line[2])


tri = np.arange(n*3).reshape((n,3))
plt.figure(figsize=(5,5))
plt.tripcolor(xy[:,:,0].ravel(), xy[:,:,1].ravel(), sol[:,:,0].ravel(), triangles=tri, cmap="turbo", shading='gouraud')
plt.triplot(xy[:,:,0].ravel(), xy[:,:,1].ravel(), triangles=tri, color="k", linewidth=0.1, alpha=0.2)
plt.axis("equal")
plt.axis("off")
plt.savefig("eta.png", dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()
plt.close()

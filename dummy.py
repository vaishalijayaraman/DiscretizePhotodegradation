img_grid = np.transpose(img)
img_grid = img / 255.0 #sets black to 0 and white to 1 

			if img_grid[i][j] != 255:
				int_star = 0

			else:
				int_star = computeIntensity(prev, j, alpha_pd_star, alpha_deg_star, x_num, prev_ci[j], prev_cd[j], prev_cf[j])

# iterate over each cell in the gel surface grid
for row in range(x_num):
	for col in range(y_num):
		# check if masked
		if img_grid[row][col] != 225:
			masked = True
		g = implicit(grid, ci_grid, cd_grid, cf_grid, t0, tf, t_step, z0, zf, z_step, alpha_pd_star, alpha_deg_star, z_num, ci, cd, cf, masked)
		grid = np.transpose(g)
		final_grid[row][col] = grid[:, t_num-1]
		print("final_grid: ", final_grid)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
x,y,z = final_grid
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, zdir='-z', c= 'red')
plt.savefig("demo.png")
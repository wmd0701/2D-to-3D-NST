# Reference: https://scipython.com/blog/poisson-disc-sampling-in-python/
import numpy as np
import matplotlib.pyplot as plt

def Poisson_disc_sampling(k = 10, r = 20, width = 360, height = 180, min_degee_dist = 25, plot = False):
    """
    2D Poisson disc sampling in some range. Sampled points are later transformed into 3D polar coordinate with
    x-coordinate as azimuth and y-coordinate as elevation. To solve the clustering problem at pole, a minimum 
    angular distance (great circle distance) is required.
    Reference: https://scipython.com/blog/poisson-disc-sampling-in-python/
    Arguments:
        k, r: arguments for traditional Poisson disc sampling
        width: range of x
        height: range of y
        min_degree_dist: minimum angular distance
        plot: whether to plot sampled points
    Returns:
        sampled points in numpy array
    """
    
    a = r/np.sqrt(2)
    nx, ny = int(width / a) + 1, int(height / a) + 1

    coords_list = [(ix, iy) for ix in range(nx) for iy in range(ny)]
    cells = {coords: None for coords in coords_list}

    def get_cell_coords(pt):
        return int(pt[0] // a), int(pt[1] // a)

    def get_neighbours(coords):
        dxdy = [(-1,-2),(0,-2),(1,-2),(-2,-1),(-1,-1),(0,-1),(1,-1),(2,-1),
                (-2,0),(-1,0),(1,0),(2,0),(-2,1),(-1,1),(0,1),(1,1),(2,1),
                (-1,2),(0,2),(1,2),(0,0)]
        neighbours = []
        for dx, dy in dxdy:
            neighbour_coords = coords[0] + dx, coords[1] + dy
            if not (0 <= neighbour_coords[0] < nx and
                    0 <= neighbour_coords[1] < ny):
                continue
            neighbour_cell = cells[neighbour_coords]
            if neighbour_cell is not None:
                neighbours.append(neighbour_cell)
        return neighbours

    def point_valid(pt):
        cell_coords = get_cell_coords(pt)
        for idx in get_neighbours(cell_coords):
            nearby_pt = samples[idx]
            distance2 = (nearby_pt[0]-pt[0])**2 + (nearby_pt[1]-pt[1])**2

            azim1 = np.deg2rad(nearby_pt[0] - 180)
            elev1 = np.deg2rad(nearby_pt[1] - 90)
            azim2 = np.deg2rad(pt[0] - 180)
            elev2 = np.deg2rad(pt[1] - 90)
            temp = np.sin(elev1)*np.sin(elev2) + np.cos(elev1)*np.cos(elev2)*np.cos(azim1-azim2)
            distance_degree = np.rad2deg(np.arccos(temp))

            if distance2 < r**2 or distance_degree < min_degee_dist:
                return False
        return True

    def get_point(k, refpt):
        i = 0
        while i < k:
            rho, theta = np.random.uniform(r, 2*r), np.random.uniform(0, 2*np.pi)
            pt = refpt[0] + rho*np.cos(theta), refpt[1] + rho*np.sin(theta)
            if not (0 <= pt[0] < width and 0 <= pt[1] < height):
                continue
            if point_valid(pt):
                return pt
            i += 1
        return False

    pt = (np.random.uniform(0, width), np.random.uniform(0, height))
    samples = [pt]
    cells[get_cell_coords(pt)] = 0
    active = [0]

    nsamples = 1
    while active:
        idx = np.random.choice(active)
        refpt = samples[idx]
        pt = get_point(k, refpt)
        if pt:
            samples.append(pt)
            nsamples += 1
            active.append(len(samples)-1)
            cells[get_cell_coords(pt)] = len(samples) - 1
        else:
            active.remove(idx)

    if(plot):
        plt.scatter(*zip(*samples), color='r', alpha=0.6, lw=0)
        plt.xlim(0, width)
        plt.ylim(0, height)
        plt.axis('off')
        plt.show()

    return np.array(samples)
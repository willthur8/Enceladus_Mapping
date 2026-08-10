import numpy as np
from smap/src/smap/random_map_+_orbit_generator.ipynb import make_map, make_sweep, coverage_report, make_perlin_splat
from scipy import optimize

# get coverage fraction from vector of parallel sweeps
def coverage_of(x, splat, domain, width):
    theta = x[0]
    offsets = x[1:]
    sweeps = [make_sweep(theta, o, width, domain) for o in offsets]
    return coverage_report(splat, sweeps, domain)["coverage_fraction"]

def greedy_seed(splat, domain, n_passes, width, n_angles=12, n_offsets=41):
    # make grid of angles and offests
    R = np.sqrt(domain.area / np.pi)
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    offset_grid = np.linspace(-R, R, n_offsets)

    best_vector, best_cov = None, -1.0
    for theta in angles:
        chosen = []
        remaining = splat
        for i in range(n_passes):
            gains = []

            for offset in offset_grid:
                sweep = make_sweep(theta, offset, width, domain)
                gains.append(remaining.intersection(sweep).area)

            offset_best = offset_grid[np.argmax(gains)]
            chosen.append(offset_best)

            remaining = remaining.difference(
                make_sweep(theta, offset_best, width, domain))
            
        vector = np.array([theta, *chosen])
        cov = coverage_of(vector, splat, domain, width)
        if cov > best_cov:
            best_vector, best_cov = vector, cov

    return best_vector, best_cov

def objective(x, splat, domain, width):
    return -coverage_of(x,splat, domain, width)

R = np.sqrt(domain.area/np.pi)
width = 0.3
K = 4 # number of passes/offsets
bounds = [(0,np.pi)] + [(-R,R)] * K

domain = make_map(1.0)

greedy_seed, greedy_coverage = greedy_seed(splat, domain, K, 0.3, n_angles=12, n_offsets=41)


result = optimize.differential_evolution(
    objective,
    bounds,
    args = [splat, domain, width],
    x0 = greedy_vec,
    popsize = 12
    maxiter = 40,
    rng = 0,
    polish= False

)

best_vector = result.x
best_coverage = -result.fun
print(best_vector, best_coverage)


        



'''
    IMPORTS
'''
#Basic package imports
import MDAnalysis as md
import pandas as pd
import numpy as np
from tqdm import tqdm

#Specifically used for grid generation
from scipy.spatial import KDTree
import itertools
from scipy.interpolate import RegularGridInterpolator

#This one is for timing the approach
from datetime import datetime

#Dolfinx stuff for current calculation
from mpi4py import MPI
from petsc4py import PETSc
import ufl

import dolfinx
import dolfinx.mesh as dmesh
import dolfinx.fem as fem

from dolfinx.fem.petsc import LinearProblem

'''
    USER DEFINED VARIABLES
'''

bulk_conductivity = 10.5 #S/m 1M KCl experimental (assuming 1 atm & ~295K)
Voltage = 0.180 #V

#Assuming your psf and pdb have the same names
system_dir = "./data/test/system_complete" 

'''
    READ SYSTEM USING MDANALYSIS
'''
system_psf = system_dir + ".psf" #For now, we consider the full system
system_pdb = system_dir + ".pdb"

system_universe = md.Universe(system_psf, system_pdb)
system = system_universe.select_atoms("not water and not name CLA POT")

'''
    GRID RELATED VARIABLES
'''
#These are from the PBC simulation I use (all in A)
size_x = 59
size_y = size_x #x & y should have same size
size_z = 85

#Atom Radii (in A)
radii_dict = dict(zip(pd.read_csv("./data/radial.csv")["Atom_type"], pd.read_csv("./data/radial.csv")["radius(A)"]))

#Radial linear conductance variables (from https://pubs.acs.org/doi/10.1021/acssensors.8b01375)
r_min = 1.4 #A
r_slope_inv = 2.9 #A (this should be divided for actual slope)

#Useful intermediate vars
r_intercept = -1.0 * r_min / r_slope_inv
r_max = r_slope_inv + r_min

#Conductivity Clippings
conductivity_min = 1e-7
conductivity_max = 1.0

#For conductivity map, just use np.clip(dist / r_slope_inv + r_intercept, conductivity_min, conductivity_max)

#Write the radii for each atom and add as attribute to 
names = np.strings.slice(system.names.astype(np.dtypes.StringDType), 0,1)
radii = np.array([radii_dict[k] for k in names])

'''
    Construct the grid and add the conductances based on the closest atom (this is using the tree query method)
'''

#Grid spacing along dimensions
x_arr = np.arange(-1.0 * int(size_x / 2), 1.0 * int(size_x / 2), 1, dtype=int)
y_arr = np.arange(-1.0 * int(size_y / 2), 1.0 * int(size_y / 2), 1, dtype=int)
z_arr = np.arange(-1.0 * int(size_z / 2), 1.0 * int(size_z / 2), 1, dtype=int)

#Method for having nice tqdm progress bar
coords = np.array(list(itertools.product(x_arr,y_arr,z_arr)))

#Generate the KDTree
tree_pos = KDTree(system.positions)
tree_grid = KDTree(coords)

#Make the query, returns list of nearest neighbor for every point 
query = tree_grid.query_ball_tree(tree_pos, r=(r_max + 1.9)) #This does all querys between the two trees

#Reformat to a numpy via a pandas dataframe for ease of use since the list is inhomogenous originally
query_np = pd.DataFrame(query).to_numpy()

#Will store final conductivity map
conductivity_map = np.empty((len(query), 1))

#For any point that has nothing within maximum distance, we set to max bulk
conductivity_map[np.all(np.isnan(query_np), axis=1)] = bulk_conductivity

#Now we deal with the ones with a maximum distance
indices=np.arange(0,query_np.shape[0]) #Used to get the indices 

#Initialize a temp copy so that we can replace used up rows with NaNs 
temp_query = query_np.copy()

#We already did the all nan case so we ignore it, look at all rows with non-all nan
for index in reversed(range(query_np.shape[1])[1:]):

    #For easier use, we use the spliced query
    temp_query = temp_query[:,:index]

    #Construct the mask to look at rows with non NaN values
    mask = np.all(~np.isnan(temp_query), axis=1)
    
    #Compute the distances from given grid coords
    dist = np.linalg.norm(system.positions[temp_query[mask].astype(int)] - coords[mask][:,None,:], axis=2)
    real_dist = dist - radii[temp_query[mask].astype(int)] #Correct the distances with the atom's surface
    
    #Grab the smallest distances
    shortest_dist = np.min(real_dist, axis=1)
    
    #Update the respective conductivities
    conductivity_map[mask] = bulk_conductivity * np.clip(np.expand_dims(shortest_dist,-1) / r_slope_inv + r_intercept, conductivity_min, conductivity_max)

    #Change the used rows to all NaNs so we don't redo already done positions while ignoring certain neighbors
    temp_query[mask] = np.nan

#Construct the interpolator
interpolator = RegularGridInterpolator((x_arr, y_arr, z_arr), conductivity_map.reshape(len(x_arr), len(y_arr), len(z_arr)), bounds_error=False, fill_value=bulk_conductivity)

'''
    Dolfinx setup following Pinhao's code
'''

#Setting up MPI communication layer
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

#Coords of lower corner and upper (respectively opposite) corner
mesh_coords = np.array([[-1.0 * size_x / 2, -1.0 * size_y / 2, -1.0 * size_z / 2],
                        [size_x / 2, size_y / 2, size_z / 2]])

#Construct dolfinx mesh
mesh = dmesh.create_box(comm, mesh_coords, [len(x_arr), len(y_arr), len(z_arr)])

#Functions for potential and conductivity 
V = fem.functionspace(mesh, ("Lagrange", 1))
sig = fem.Function(V)
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

#Boundary condition 
fdim = mesh.topology.dim -1
tol = 1e-14
zmin = -1.0 * size_z / 2.0
zmax = size_z / 2.0

top_facets = dmesh.locate_entities_boundary(mesh, fdim, lambda x: np.abs(x[2] - zmax) < tol)
bot_facets = dmesh.locate_entities_boundary(mesh, fdim, lambda x: np.abs(x[2] - zmin) < tol)

top_dofs = fem.locate_dofs_topological(V, fdim, top_facets)
bot_dofs = fem.locate_dofs_topological(V, fdim, bot_facets)
bc_top = fem.dirichletbc(PETSc.ScalarType(Voltage), top_dofs, V)
bc_bot = fem.dirichletbc(PETSc.ScalarType(0.0),  bot_dofs, V)
bcs = [bc_top, bc_bot]

# Facet tags for flux integration
facet_indices = np.concatenate([top_facets, bot_facets])
facet_values = np.concatenate([np.full_like(top_facets, 1, dtype=np.int32), np.full_like(bot_facets, 2, dtype=np.int32),])
facet_tag = dmesh.meshtags(mesh, fdim, facet_indices, facet_values)

# Variational form: ∫ σ ∇u·∇v dx = 0
a = sig * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = fem.Constant(mesh, PETSc.ScalarType(0.0)) * v * ufl.dx

# Measure for boundary integrals
ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag)
fixed_normal = ufl.as_vector((0.0, 0.0, 1.0))  # to match the original Constant((0,0,1))

# Solver options (to match your GMRES + AMG settings)
petsc_opts = {
    "ksp_type": "gmres",
    "pc_type": "hypre",
    "ksp_rtol": 1e-8,
    "ksp_max_it": 20000,
}

#Generate sig from the conductivity map using interpolation
grid_coords = V.tabulate_dof_coordinates()
grid_values = interpolator(grid_coords)
sig.x.array[:] = grid_values
sig.x.scatter_forward()

'''
    Computing the SEM current
'''

#Solve problem
problem = LinearProblem(a, L, bcs=bcs, petsc_options=petsc_opts, petsc_options_prefix="sem_")
solution = problem.solve()

#Compute the flux
flux_top_form = ufl.dot(fixed_normal, sig * ufl.grad(solution)) * ds(1)
flux_bot_form = ufl.dot(fixed_normal, sig * ufl.grad(solution)) * ds(2)
ft_local = fem.assemble_scalar(fem.form(flux_top_form))
fb_local = fem.assemble_scalar(fem.form(flux_bot_form))

# Sum across ranks
ft = comm.allreduce(ft_local, op=MPI.SUM)
fb = comm.allreduce(fb_local, op=MPI.SUM)

print(f"Current: {ft}")
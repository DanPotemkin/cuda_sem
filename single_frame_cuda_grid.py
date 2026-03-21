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

#CUDA Torch stuff
import torch
from torch_kdtree import build_kd_tree

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

device = 0
k = 10

#Grid spacing along dimensions
x_arr = torch.arange(-1.0 * int(size_x / 2), 1.0 * int(size_x / 2), 1, dtype=int).to(device)
y_arr = torch.arange(-1.0 * int(size_y / 2), 1.0 * int(size_y / 2), 1, dtype=int).to(device)
z_arr = torch.arange(-1.0 * int(size_z / 2), 1.0 * int(size_z / 2), 1, dtype=int).to(device)

coords_x, coords_y, coords_z = torch.meshgrid(x_arr, y_arr, z_arr, indexing='ij')

size_x = coords_x.shape[0]
size_y = coords_y.shape[1]
size_z = coords_z.shape[2]

coords = torch.concatenate([coords_x.unsqueeze(3), coords_y.unsqueeze(3), coords_z.unsqueeze(3)], axis=3)
coords = coords.reshape(size_x * size_y * size_z, 3)

system_positions = torch.as_tensor(system.positions, device=device)

tree_system = build_kd_tree(system_positions)
distances, indices = tree_system.query(coords.to(torch.float32), nr_nns_searches=k)

radii_torch = torch.as_tensor(radii, device=device)

conductivity_map = torch.min(torch.sqrt(distances) - radii_torch[indices], axis=1).values
conductivity_map = bulk_conductivity * torch.clamp(conductivity_map / r_slope_inv + r_intercept, 0, 1)

x_arr_cpu = x_arr.detach().cpu().numpy()
y_arr_cpu = y_arr.detach().cpu().numpy()
z_arr_cpu = z_arr.detach().cpu().numpy()

conductivity_map_cpu = conductivity_map.detach().cpu().numpy()

interpolator = RegularGridInterpolator((x_arr_cpu, y_arr_cpu, z_arr_cpu), conductivity_map_cpu.reshape(size_x, size_y, size_z), bounds_error=False, fill_value=bulk_conductivity)

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
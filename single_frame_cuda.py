'''
    Imports
'''

import MDAnalysis as md
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import itertools
from tqdm import tqdm

from scipy.interpolate import RegularGridInterpolator

#This one is for timing the approach
from datetime import datetime

#Dolfinx stuff for current calculation
from mpi4py import MPI
from petsc4py import PETSc
import ufl

#Import cudolfinx
import cudolfinx as cufem

#Need several things from original dolfinx as well
from dolfinx import mesh as dmesh, fem
from dolfinx.fem import petsc as fe_petsc

'''
    Setup the md analysis universe
'''

system_psf = "data/test/system_complete.psf" #For now, we consider the full system
system_pdb = "data/test/system_complete.pdb"

system_universe = md.Universe(system_psf, system_pdb)
system = system_universe.select_atoms("not water and not name CLA POT")

'''
    Grid related variables
'''

bulk_conductivity = 10.5 #S/m 1M KCl experimental (assuming 1 atm & ~295K)

#These are from the PBC simulation I use (all in A)
size_x = 59
size_y = size_x #x & y should have same size
size_z = 85

#Atom Radii (in A)
radii_dict = dict(zip(pd.read_csv("data/radial.csv")["Atom_type"], pd.read_csv("data/radial.csv")["radius(A)"]))

#Radial linear conductance variables (from https://pubs.acs.org/doi/10.1021/acssensors.8b01375)
r_min = 1.4 #A
r_slope_inv = 2.9 #A (this should be divided for actual slope)

#Useful intermediate vars
r_intercept = -1.0 * r_min / r_slope_inv
r_max = r_slope_inv + r_min

#For conductivity map, just use np.clip(dist / r_slope_inv + r_intercept, 0, 1)

#Write the radii for each atom and add as attribute to 
names = np.strings.slice(system.names.astype(np.dtypes.StringDType), 0,1)
radii = np.array([radii_dict[k] for k in names])

'''
    Possibly faster scipy KD-Tree
'''

#Grid spacing along dimensions
x_arr = np.arange(-1.0 * int(size_x / 2), 1.0 * int(size_x / 2), 1, dtype=int)
y_arr = np.arange(-1.0 * int(size_y / 2), 1.0 * int(size_y / 2), 1, dtype=int)
z_arr = np.arange(-1.0 * int(size_z / 2), 1.0 * int(size_z / 2), 1, dtype=int)

coords_x, coords_y, coords_z = np.meshgrid(x_arr, y_arr, z_arr, indexing='ij')

size_x = coords_x.shape[0]
size_y = coords_y.shape[1]
size_z = coords_z.shape[2]

coords = np.concatenate([np.expand_dims(coords_x,-1), np.expand_dims(coords_y, -1), np.expand_dims(coords_z, -1)], axis=3)
coords = coords.reshape(size_x * size_y * size_z, 3)

tree_pos = KDTree(system.positions)

distances, indices = tree_pos.query(coords, k=10)
conductivity_map = np.min(distances - radii[indices], axis=1)
conductivity_map = bulk_conductivity * np.clip(conductivity_map / r_slope_inv + r_intercept, 0, 1)

interpolator = RegularGridInterpolator((x_arr, y_arr, z_arr), conductivity_map.reshape(size_x, size_y, size_z), bounds_error=False, fill_value=bulk_conductivity)

'''
    Setting up Mesh
'''

#Setting up MPI communication layer
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

#Coords of lower corner and upper (respectively opposite) corner
mesh_coords = np.array([[-1.0 * size_x / 2, -1.0 * size_y / 2, -1.0 * size_z / 2],
                        [size_x / 2, size_y / 2, size_z / 2]])

#Construct dolfinx mesh
mesh = dmesh.create_box(comm, mesh_coords, [len(x_arr), len(y_arr), len(z_arr)])

'''
    Set up Trial Functions and BC
'''

#Problem variables
Voltage = .180 #Volts

#Functions for potential and conductivity 
V = fem.functionspace(mesh, ("Lagrange", 1))
sig = fem.Function(V)
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

#Boundary conditions
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

# Measure for boundary integrals
ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag)
fixed_normal = ufl.as_vector((0.0, 0.0, 1.0))

#Generate sig from the conductivity map using interpolation
grid_coords = V.tabulate_dof_coordinates()
grid_values = interpolator(grid_coords)
sig.x.array[:] = grid_values
sig.x.scatter_forward()

'''
    Setup Variational Problem and Send to CUDA Assembler
'''

# Variational form: ∫ σ ∇u·∇v dx = 0
a = sig * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = fem.Constant(mesh, PETSc.ScalarType(0.0)) * v * ufl.dx

#Generate CUDA counterparts for variational form
a = cufem.form(a)
L = cufem.form(L)
asm = cufem.CUDAAssembler()
cuda_A = asm.create_matrix(a)
cuda_b = asm.create_vector(L)
b = cuda_b.vector

#Generate BCs for CUDA assembler
device_bcs = asm.pack_bcs(bcs)

#Assemble the matrices and vectors on the cuda assembly
asm.assemble_matrix(a, cuda_A, bcs=device_bcs)
cuda_A.assemble()
A = cuda_A.mat

asm.assemble_vector(L, cuda_b)
asm.apply_lifting(cuda_b, [a], [device_bcs])
asm.set_bc(cuda_b, bcs=device_bcs, V=V)

'''
    Setup & solve the problem
'''

ksp = PETSc.KSP().create(comm)
ksp.setOperators(A)
ksp.setType("gmres")
pc = ksp.getPC()
pc.setType("jacobi")

solution = b.copy()
ksp.solve(b, solution)

#Transfer solution back to cpu
solution_cpu = fem.Function(V)
solution_cpu.x.array[:] = solution.array[:]

'''
    Flux calculation
'''

#Compute the flux
flux_top_form = ufl.dot(fixed_normal, sig * ufl.grad(solution_cpu)) * ds(1)
flux_bot_form = ufl.dot(fixed_normal, sig * ufl.grad(solution_cpu)) * ds(2)
ft_local = fem.assemble_scalar(fem.form(flux_top_form))
fb_local = fem.assemble_scalar(fem.form(flux_bot_form))

# Sum across ranks
ft = comm.allreduce(ft_local, op=MPI.SUM)
fb = comm.allreduce(fb_local, op=MPI.SUM)

print(f"Current: {ft}")
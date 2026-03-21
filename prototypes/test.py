from mpi4py import MPI
from dolfinx import fem , mesh
import cudolfinx as cufem
from ufl import dx , inner , grad
import ufl

N = 1000
domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N)
V = fem.functionspace(domain , ("Lagrange", 1))
f = fem.Function (V)
f. interpolate(lambda x: x[0] ** 2 + x[1])
u, v = ufl.TestFunction(V), ufl.TrialFunction(V)
A = -inner(grad(u), grad(v)) * dx
L = f * v * ufl.dx

cuda_A = cufem.form(A)
cuda_L = cufem.form(L)
asm = cufem.CUDAAssembler()
mat = asm.assemble_matrix(cuda_A)
vec = asm.assemble_vector(cuda_L)

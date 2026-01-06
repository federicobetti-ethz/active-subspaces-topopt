"""Sample script to compute the active subspaces of a topology optimization problem."""

import dolfinx
import ufl
from mpi4py import MPI
import basix
import numpy as np

from dolfinx.fem.petsc import LinearProblem, assemble_matrix, assemble_vector, create_matrix, create_vector
from petsc4py import PETSc

def main():
    n = 20
    mesh = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [2.0, 1.0]], [2 * n, n], dolfinx.mesh.CellType.quadrilateral)

    de = basix.ufl.element("CG", mesh.basix_cell(), 1, shape=(2,))
    V = dolfinx.fem.functionspace(mesh, de)

    rhofe = basix.ufl.element("CG", mesh.basix_cell(), 1)
    Q = dolfinx.fem.functionspace(mesh, rhofe)

    rhoe = basix.ufl.element("DG", mesh.basix_cell(), 0)
    W = dolfinx.fem.functionspace(mesh, rhoe)

    rho = dolfinx.fem.Function(W)
    rho_f = dolfinx.fem.Function(Q)

    u = dolfinx.fem.Function(V)

    E0, Emin = 1.0, 1e-6
    penalty = 3.0

    mu = 1.0
    lmbda = 1.0

    def eps(v):
        return ufl.sym(ufl.grad(v))

    def sigma(v, rho_f):
        E = Emin + (E0 - Emin) * rho_f ** penalty
        return E * (2 * mu * eps(v) + lmbda * ufl.tr(eps(v)) * ufl.Identity(mesh.topology.dim))

    u_trial = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(sigma(u_trial, rho_f), eps(v)) * ufl.dx

    left_facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1,
                                                lambda x: np.isclose(x[0], 0.0))
    left_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, left_facets)
    bc = dolfinx.fem.dirichletbc(np.zeros(mesh.topology.dim), left_dofs, V)

    right_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[0], 2.0)
    )
    facet_tags = dolfinx.mesh.meshtags(mesh, mesh.topology.dim - 1, right_facets, np.full(len(right_facets), 1, dtype=np.int32))
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
    load_expr = dolfinx.fem.Constant(mesh, (0.0, -10.0))
    L = ufl.inner(load_expr, v) * ds(1)
    
    A = create_matrix(dolfinx.fem.form(a))
    assemble_matrix(A, dolfinx.fem.form(a), bcs=[bc])
    A.assemble()

    b = create_vector(dolfinx.fem.form(L))
    assemble_vector(b, dolfinx.fem.form(L))
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.apply_lifting(b, [dolfinx.fem.form(a)], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b, [bc])

    ksp = PETSc.KSP().create(MPI.COMM_WORLD)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    
    r_min = 0.1
    wf = ufl.CellDiameter(mesh)
    r = r_min * wf

    rho_f_trial = ufl.TrialFunction(Q)
    w = ufl.TestFunction(Q)
    filter_form = (r**2 * ufl.inner(ufl.grad(rho_f_trial), ufl.grad(w)) +
                rho_f_trial * w) * ufl.dx
    filter_rhs = rho * w * ufl.dx

    A_filter = create_matrix(dolfinx.fem.form(filter_form))
    b_filter = create_vector(dolfinx.fem.form(filter_rhs))
    assemble_matrix(A_filter, dolfinx.fem.form(filter_form))
    A_filter.assemble()

    filter_ksp = PETSc.KSP().create(MPI.COMM_WORLD)
    filter_ksp.setOperators(A_filter)
    filter_ksp.setFromOptions()

    def apply_filter(rho, rho_f):
        b_filter.zeroEntries()
        assemble_vector(b_filter, dolfinx.fem.form(filter_rhs))
        b_filter.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        filter_ksp.solve(b_filter, rho_f.x.petsc_vec)
        rho_f.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)

    w_W = ufl.TestFunction(W)
    
    def apply_adjoint_filter(grad_Q_vec, grad_W_vec):
        temp = dolfinx.fem.Function(Q)
        filter_ksp.solve(grad_Q_vec, temp.x.petsc_vec)
        temp.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        
        adjoint_rhs_form = temp * w_W * ufl.dx
        b_adjoint = create_vector(dolfinx.fem.form(adjoint_rhs_form))
        assemble_vector(b_adjoint, dolfinx.fem.form(adjoint_rhs_form))
        b_adjoint.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        
        grad_W_vec.copy(b_adjoint)
        grad_W_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    J_form = ufl.inner(load_expr, u) * ds(1)

    w_Q = ufl.TestFunction(Q)
    dE_drho_f = penalty * (E0 - Emin) * rho_f ** (penalty - 1)
    E_current = Emin + (E0 - Emin) * rho_f ** penalty
    strain_energy_density = ufl.inner(sigma(u, rho_f), eps(u))
    DJ_form = -(dE_drho_f / E_current) * strain_energy_density * w_Q * ufl.dx

    def objective(rho_vec):
        apply_filter(rho, rho_f)
        A.zeroEntries()
        assemble_matrix(A, dolfinx.fem.form(a), bcs=[bc])
        A.assemble()
        b.zeroEntries()
        assemble_vector(b, dolfinx.fem.form(L))
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.apply_lifting(b, [dolfinx.fem.form(a)], [[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.set_bc(b, [bc])
        ksp.setOperators(A)
        ksp.solve(b, u.x.petsc_vec)
        u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        return dolfinx.fem.assemble_scalar(dolfinx.fem.form(J_form))

    def gradient(rho_vec, grad_out, debug=False):
        apply_filter(rho, rho_f)
        
        A.zeroEntries()
        assemble_matrix(A, dolfinx.fem.form(a), bcs=[bc])
        A.assemble()
        
        b.zeroEntries()
        assemble_vector(b, dolfinx.fem.form(L))
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.apply_lifting(b, [dolfinx.fem.form(a)], [[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        
        dolfinx.fem.set_bc(b, [bc])
        ksp.setOperators(A)
        ksp.solve(b, u.x.petsc_vec)
        
        u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        
        psi_Q_trial = ufl.TrialFunction(Q)
        w_Q = ufl.TestFunction(Q)
        strain_energy_expr = ufl.inner(sigma(u, rho_f), eps(u))
        
        A_psi = create_matrix(dolfinx.fem.form(psi_Q_trial * w_Q * ufl.dx))
        assemble_matrix(A_psi, dolfinx.fem.form(psi_Q_trial * w_Q * ufl.dx))
        A_psi.assemble()
        b_psi = create_vector(dolfinx.fem.form(strain_energy_expr * w_Q * ufl.dx))
        assemble_vector(b_psi, dolfinx.fem.form(strain_energy_expr * w_Q * ufl.dx))
        b_psi.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        
        psi_Q = dolfinx.fem.Function(Q)
        ksp_psi = PETSc.KSP().create(MPI.COMM_WORLD)
        ksp_psi.setOperators(A_psi)
        ksp_psi.setType("preonly")
        ksp_psi.getPC().setType("lu")
        ksp_psi.solve(b_psi, psi_Q.x.petsc_vec)
        psi_Q.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        
        dE_drho_f = penalty * (E0 - Emin) * rho_f ** (penalty - 1)
        DJ_form_current = -dE_drho_f * psi_Q * w_Q * ufl.dx
        
        grad_Q = create_vector(dolfinx.fem.form(DJ_form_current))
        grad_Q.zeroEntries()
        assemble_vector(grad_Q, dolfinx.fem.form(DJ_form_current))
        grad_Q.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE
        )
        
        grad_Q_func = dolfinx.fem.Function(Q)
        grad_Q_func.x.petsc_vec.copy(grad_Q)
        grad_Q_func.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        
        if debug:
            grad_Q_array = grad_Q_func.x.array
            print(f"  Grad_Q min: {np.min(grad_Q_array):.6e}, max: {np.max(grad_Q_array):.6e}, mean: {np.mean(grad_Q_array):.6e}")
        
        apply_adjoint_filter(grad_Q_func.x.petsc_vec, grad_out)

    max_it = 50
    vol_frac = 0.4
    rho_MIN = 1e-3

    file = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "topopt2d.xdmf", "w")
    file.write_mesh(mesh)

    rho.x.array[:] = 0.5
    rho.name = "Density"
    u.name = "Displacement"

    for it in range(max_it):
        obj = objective(rho.x.array)
        grad = dolfinx.fem.Function(W)
        gradient(rho.x.array, grad.x.petsc_vec, it == 0)
        
        if it == 0:
            print(f"  Grad min: {np.min(grad.x.array):.6e}, max: {np.max(grad.x.array):.6e}, mean: {np.mean(grad.x.array):.6e}")

        l1, l2 = 0, 1e9
        move = 0.2
        while (l2 - l1) > 1e-6:
        
            lmid = 0.5 * (l1 + l2)
            grad_safe = np.minimum(grad.x.array, -1e-12)
            new_rho = np.maximum(rho_MIN,
                        np.maximum(rho.x.array - move,
                        np.minimum(1.0,
                        np.minimum(rho.x.array + move,
                        rho.x.array * np.sqrt(-grad_safe / lmid)))))
        
            if np.mean(new_rho) - vol_frac > 0:
                l2 = lmid
            else:
                l1 = lmid
        
        rho.x.array[:] = new_rho

        file.write_function(rho, t=it)
        file.write_function(u, t=it)

        print(f"Iter {it+1:3d}, Obj {obj:.6e}, Vol {np.mean(rho.x.array):.3f}")

    file.close()

if __name__ == "__main__":
    main()
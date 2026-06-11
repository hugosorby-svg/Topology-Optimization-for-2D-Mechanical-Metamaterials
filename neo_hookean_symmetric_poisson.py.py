from dolfin import *
from dolfin_adjoint import *
import ufl
import numpy as np
import os
from time import strftime

parameters["std_out_all_processes"] = False
parameters["form_compiler"]["quadrature_degree"] = 2 if os.getenv("TOP_FAST", "0") == "1" else 4

solver_parameters = {
    "linear_solver": "mumps",
    "preconditioner": "default",
}

# -------------------------
# Global setup
# -------------------------
FAST = os.getenv("TOP_FAST", "0") == "1"
Lx, Ly = 1.0, 1.0
nelx, nely = (48, 48) if FAST else (96, 96)

volfrac = 0.40
rho_min = 1e-3
penal = 3.0
r_filter = 0.03
beta_proj = 8.0
eta_proj = 0.5

# -------------------------
# User options
# -------------------------
OBJECTIVE_MODE = "poisson_ratio"
TARGET_POISSON_RATIO = -0.80
ORTHO_PROFILE = "x"

MODE_OPTIONS = {
    "1": "poisson_ratio",
    "2": "orthotropic",
    "3": "shear_stiff",
}

ORTHO_OPTIONS = {
    "1": "x",
    "2": "y",
    "3": "xy",
}


def read_optional_float(prompt, default=None):
    raw = input(prompt).strip()
    if raw == "":
        return default
    return float(raw)


def read_menu_choice(prompt, options, default_value):
    reverse_lookup = dict((value, key) for key, value in options.items())
    choice = input("%s [default: %s]: " % (prompt, reverse_lookup[default_value])).strip()
    if choice == "":
        return default_value
    if choice in options:
        return options[choice]
    return default_value


def configure_objective_from_prompt():
    global OBJECTIVE_MODE, TARGET_POISSON_RATIO, ORTHO_PROFILE

    if os.getenv("TOP_INTERACTIVE", "1") != "1":
        return

    print("")
    print("Choose optimization mode:")
    print("  1: poisson_ratio")
    print("  2: orthotropic")
    print("  3: shear_stiff")
    OBJECTIVE_MODE = read_menu_choice("Enter 1, 2 or 3", MODE_OPTIONS, OBJECTIVE_MODE)

    if OBJECTIVE_MODE == "poisson_ratio":
        TARGET_POISSON_RATIO = read_optional_float(
            "Target Poisson ratio [default: %.6f]: " % TARGET_POISSON_RATIO,
            TARGET_POISSON_RATIO,
        )
    elif OBJECTIVE_MODE == "orthotropic":
        print("Choose beam direction profile:")
        print("  1: x")
        print("  2: y")
        print("  3: xy")
        ORTHO_PROFILE = read_menu_choice("Enter 1, 2 or 3", ORTHO_OPTIONS, ORTHO_PROFILE)


configure_objective_from_prompt()

# Objective weights
w_nu = 1.0
w_stiff = 2.0
w_vol = 8.0
w_xy_balance = 2.5
w_xy_min_stiff = 8.0
w_xy_gray = 0.12
w_xy_geo = 0.02
w_xy_perimeter = 0.015
w_xy_centering = 0.20
c_min = 0.08
smooth_k = 40.0
macro_eps = 1.0e-2
bin_threshold = 0.5

# Large-strain verification setup
SET_DEFORMATION = 0.25
lambda2_max = 1.0 + SET_DEFORMATION

# Base material parameters
E0 = 1.0
nu0 = 0.30
mu0 = E0 / (2.0 * (1.0 + nu0))
lmbda0 = E0 * nu0 / ((1.0 + nu0) * (1.0 - 2.0 * nu0))

# Continuation schedule
continuation_stages = [
    (1.5, 2.0, 12 if FAST else 30),
    (2.5, 4.0, 14 if FAST else 35),
    (3.5, 8.0, 16 if FAST else 45),
]


class PeriodicBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool(
            on_boundary
            and ((near(x[0], 0.0) and (not near(x[1], Ly)))
                 or (near(x[1], 0.0) and (not near(x[0], Lx))))
        )

    def map(self, x, y):
        if near(x[0], Lx) and near(x[1], Ly):
            y[0] = x[0] - Lx
            y[1] = x[1] - Ly
        elif near(x[0], Lx):
            y[0] = x[0] - Lx
            y[1] = x[1]
        else:
            y[0] = x[0]
            y[1] = x[1] - Ly


mesh = RectangleMesh(MPI.comm_world, Point(0.0, 0.0), Point(Lx, Ly), nelx, nely)
pbc = PeriodicBoundary()
V = VectorFunctionSpace(mesh, "CG", 1, constrained_domain=pbc)
Q = FunctionSpace(mesh, "CG", 1, constrained_domain=pbc)
Ve = VectorElement("CG", mesh.ufl_cell(), 1)
Re = FiniteElement("R", mesh.ufl_cell(), 0)
W = FunctionSpace(mesh, MixedElement([Ve, Re]), constrained_domain=pbc)

x = SpatialCoordinate(mesh)
I = Identity(2)
cell_volume = assemble(Constant(1.0) * dx(domain=mesh))


# -------------------------
# Density and material interpolation
# -------------------------
def simp_theta(rho_phys):
    return rho_min + (1.0 - rho_min) * rho_phys**penal


def filter_density(rho):
    rhof = Function(Q, name="rho_tilde")
    v = TrialFunction(Q)
    w = TestFunction(Q)
    a = (r_filter * r_filter * dot(grad(v), grad(w)) + v * w) * dx
    L = rho * w * dx
    solve(a == L, rhof, solver_parameters=solver_parameters)
    return rhof


def project_density(rho_tilde):
    den = np.tanh(beta_proj * eta_proj) + np.tanh(beta_proj * (1.0 - eta_proj))
    num = np.tanh(beta_proj * eta_proj) + ufl.tanh(beta_proj * (rho_tilde - eta_proj))
    return num / den


def psi_neo(F, rho_phys):
    theta = simp_theta(rho_phys)
    mu = mu0 * theta
    lmbda = lmbda0 * theta
    J = det(F)
    Ic = tr(F.T * F)
    return 0.5 * mu * (Ic - 2.0) - mu * ln(J) + 0.5 * lmbda * ln(J) ** 2


def first_piola(F, rho_phys):
    Fv = variable(F)
    return diff(psi_neo(Fv, rho_phys), Fv)


# -------------------------
# Homogenization and effective stiffness
# -------------------------
def solve_case(rho_phys, eps_xx, eps_yy, gamma_xy, name):
    # Solve one unit-cell problem for one explicit macro strain state.
    w = Function(V, name=name)
    v = TestFunction(V)
    du = TrialFunction(V)

    ubar = as_vector((
        eps_xx * x[0] + gamma_xy * x[1],
        gamma_xy * x[0] + eps_yy * x[1],
    ))
    F = I + grad(w + ubar)
    P = first_piola(F, rho_phys)

    R = inner(P, grad(v)) * dx
    J = derivative(R, w, du)

    pin = DirichletBC(V, Constant((0.0, 0.0)), "near(x[0], 0.0) && near(x[1], 0.0)", "pointwise")
    problem = NonlinearVariationalProblem(R, w, [pin], J)
    solver = NonlinearVariationalSolver(problem)
    solver.parameters.update({"nonlinear_solver": "newton", "newton_solver": {
        "linear_solver": "mumps",
        "relative_tolerance": 1e-10,
        "absolute_tolerance": 1e-12,
        "maximum_iterations": 20,
        "error_on_nonconvergence": True,
    }})
    solver.solve()
    return w, P


def compute_effective_stiffness_2d(rho_phys, case_prefix):
    # Solve the three basic 2D load cases:
    # case 11 -> [eps_xx, eps_yy, gamma_xy] = [macro_eps, 0, 0]
    # case 22 -> [eps_xx, eps_yy, gamma_xy] = [0, macro_eps, 0]
    # case 12 -> [eps_xx, eps_yy, gamma_xy] = [0, 0, macro_eps]

    w_11, P_11 = solve_case(rho_phys, macro_eps, 0.0, 0.0, "%s_11" % case_prefix)
    w_22, P_22 = solve_case(rho_phys, 0.0, macro_eps, 0.0, "%s_22" % case_prefix)
    w_12, P_12 = solve_case(rho_phys, 0.0, 0.0, macro_eps, "%s_12" % case_prefix)

    sigma_11_xx = assemble(P_11[0, 0] * dx) / cell_volume
    sigma_11_yy = assemble(P_11[1, 1] * dx) / cell_volume
    sigma_11_xy = assemble(P_11[0, 1] * dx) / cell_volume

    sigma_22_xx = assemble(P_22[0, 0] * dx) / cell_volume
    sigma_22_yy = assemble(P_22[1, 1] * dx) / cell_volume
    sigma_22_xy = assemble(P_22[0, 1] * dx) / cell_volume

    sigma_12_xx = assemble(P_12[0, 0] * dx) / cell_volume
    sigma_12_yy = assemble(P_12[1, 1] * dx) / cell_volume
    sigma_12_xy = assemble(P_12[0, 1] * dx) / cell_volume

    # Column 1: response to eps_xx
    C11 = sigma_11_xx / macro_eps
    C21 = sigma_11_yy / macro_eps
    C61 = sigma_11_xy / macro_eps

    # Column 2: response to eps_yy
    C12 = sigma_22_xx / macro_eps
    C22 = sigma_22_yy / macro_eps
    C62 = sigma_22_xy / macro_eps

    # Column 3: response to gamma_xy
    C16 = sigma_12_xx / macro_eps
    C26 = sigma_12_yy / macro_eps
    C66 = sigma_12_xy / macro_eps


    C = [
        [C11, C12, C16],
        [C21, C22, C26],
        [C61, C62, C66],
    ]

    displacements = {
        "11": w_11,
        "22": w_22,
        "12": w_12,
    }
    return C, displacements


def poisson_objective(C11, C12, C21, C22):
    # Poisson mode: match the chosen effective Poisson ratio.
    nu_eff = effective_poisson_response(C11, C12, C21, C22)
    aniso = ((C11 - C22) / (C11 + C22 + 1e-8)) ** 2
    stiff_x_barrier = ln(1.0 + exp(smooth_k * (c_min - C11))) / smooth_k
    stiff_y_barrier = ln(1.0 + exp(smooth_k * (c_min - C22))) / smooth_k
    stiffness_penalty = w_stiff * (stiff_x_barrier ** 2 + stiff_y_barrier ** 2)
    return (
        w_nu * (nu_eff - TARGET_POISSON_RATIO) ** 2
        + 1.5 * aniso
        + stiffness_penalty
    )


def effective_poisson_response(C11, C12, C21, C22):
    return (0.5 * (C12 + C21)) / (C22 + 1e-8)


def shear_objective(C66, rho_phys):
    # Shear mode: maximize C66 by minimizing -C66.
    vol_phys = assemble(rho_phys * dx) / cell_volume
    vol_penalty = (vol_phys - volfrac) ** 2
    return -C66 + w_vol * vol_penalty


def orthotropic_objective(C11, C12, C21, C22, C16, C26, C61, C62, rho_phys, profile):
    # Orthotropic mode written in the simplest direct form.
    coupling_penalty = C16 ** 2 + C26 ** 2 + C61 ** 2 + C62 ** 2
    vol_phys = assemble(rho_phys * dx) / cell_volume
    vol_penalty = (vol_phys - volfrac) ** 2

    if profile == "x":
        # x mode: maximize C11
        return -(C11 - 0.3 * C22) + coupling_penalty + w_vol * vol_penalty
    
    elif profile == "y":
        # y mode: maximize C22
        return -(C22 - 0.3 * C11) + coupling_penalty + w_vol * vol_penalty
    
    elif profile == "xy":
        # xy mode: maximize C11 + C22 while keeping C11 ~= C22
        gray_penalty = assemble((rho_phys * (1.0 - rho_phys)) * dx) / cell_volume
        perimeter_penalty = assemble(dot(grad(rho_phys), grad(rho_phys)) * dx) / cell_volume
        grad_x = assemble((rho_phys.dx(0) ** 2) * dx) / cell_volume
        grad_y = assemble((rho_phys.dx(1) ** 2) * dx) / cell_volume
        geo_direction_penalty = ((grad_x - grad_y) / (grad_x + grad_y + 1e-8)) ** 2
        mass = assemble(rho_phys * dx) + 1e-8
        x_center = assemble(rho_phys * x[0] * dx) / mass
        y_center = assemble(rho_phys * x[1] * dx) / mass
        center_penalty = ((x_center - 0.5 * Lx) / Lx) ** 2 + ((y_center - 0.5 * Ly) / Ly) ** 2
        bias_reg = C11 + C22 + 1e-8
        balance_penalty = ((C11 - C22) / bias_reg) ** 2
        symmetry_penalty = (C12 - C21) ** 2
        stiff_x_barrier = ln(1.0 + exp(smooth_k * (c_min - C11))) / smooth_k
        stiff_y_barrier = ln(1.0 + exp(smooth_k * (c_min - C22))) / smooth_k
        min_stiffness_objective = stiff_x_barrier ** 2 + stiff_y_barrier ** 2
        return (
            w_xy_balance * balance_penalty
            + w_xy_min_stiff * min_stiffness_objective
            + coupling_penalty
            + symmetry_penalty
            + w_xy_gray * gray_penalty
            + w_xy_geo * geo_direction_penalty
            + w_xy_perimeter * perimeter_penalty
            + w_xy_centering * center_penalty
            + w_vol * vol_penalty
        )
    else:
        raise ValueError("Unknown ORTHO_PROFILE: %s" % profile)


def build_objective_from_mode(C, rho_phys, mode, profile):
    # Read the effective 2D Voigt matrix directly.
    C11, C12, C16 = C[0]
    C21, C22, C26 = C[1]
    C61, C62, C66 = C[2]

    # Keep a named dictionary only for reporting and for the rest of the script.
    props = {
        "C11": C11,
        "C12": C12,
        "C16": C16,
        "C21": C21,
        "C22": C22,
        "C26": C26,
        "C61": C61,
        "C62": C62,
        "C66": C66,
    }

    # Choose the objective for the selected mode.
    if mode == "poisson_ratio":
        objective = poisson_objective(C11, C12, C21, C22)
        summary = "nu=%.6f" % float(effective_poisson_response(C11, C12, C21, C22))
    elif mode == "shear_stiff":
        objective = shear_objective(C66, rho_phys)
        summary = "shear, C66=%.6f" % float(C66)
    elif mode == "orthotropic":
        objective = orthotropic_objective(C11, C12, C21, C22, C16, C26, C61, C62, rho_phys, profile)
        summary = "profile=%s" % profile
    else:
        raise ValueError("Unknown OBJECTIVE_MODE: %s" % mode)

    return objective, props, summary


def compute_gray_density_metric(rho_phys):
    return assemble((rho_phys * (1.0 - rho_phys)) * dx) / cell_volume


def print_effective_stiffness_matrix(C):
    print("Final homogenized stiffness matrix C_bar:")
    for row in C:
        print("  [{: .6e} {: .6e} {: .6e}]".format(float(row[0]), float(row[1]), float(row[2])))


def print_convergence_history(stage_history):
    print("Convergence history (post-stage):")
    for entry in stage_history:
        print(
            "  Stage {stage}: objective={objective:.6e}, nu_obj={nu_obj:.6f}, vol={vol:.6f}, "
            "gray={gray:.6e}, C11={C11:.6f}, C22={C22:.6f}, C66={C66:.6f}".format(**entry)
        )


# -------------------------
# Large-strain verification
# -------------------------
def macro_u_stretch(lambda_x, lambda_y):
    return as_vector(((lambda_x - 1.0) * x[0], (lambda_y - 1.0) * x[1]))


def solve_uniaxial_y_mixed(rho_phys, lambda2, name, z_init=None):
    z = Function(W, name=name)
    if z_init is not None:
        z.assign(z_init)

    w, alpha = split(z)
    v, q = TestFunctions(W)
    dz = TrialFunction(W)

    lambda1 = 1.0 + alpha
    ubar = macro_u_stretch(lambda1, lambda2)
    F = I + grad(w + ubar)
    P = first_piola(F, rho_phys)

    Res = inner(P, grad(v)) * dx + q * P[0, 0] * dx
    Jac = derivative(Res, z, dz)

    pin = DirichletBC(W.sub(0), Constant((0.0, 0.0)),
                      "near(x[0], 0.0) && near(x[1], 0.0)", "pointwise")
    problem = NonlinearVariationalProblem(Res, z, [pin], Jac)
    solver = NonlinearVariationalSolver(problem)
    solver.parameters.update({"nonlinear_solver": "newton", "newton_solver": {
        "linear_solver": "mumps",
        "relative_tolerance": 1e-7,
        "absolute_tolerance": 1e-9,
        "maximum_iterations": 40,
        "relaxation_parameter": 0.8,
        "error_on_nonconvergence": False,
    }})
    solver.solve()
    return z, lambda1


def initialize_density():
    rho = interpolate(Constant(volfrac), Q)
    np.random.seed(4)
    values = rho.vector().get_local()
    coords = Q.tabulate_dof_coordinates().reshape((-1, 2))
    seed = np.cos(2.0 * np.pi * 4.0 * coords[:, 0]) * np.cos(2.0 * np.pi * 4.0 * coords[:, 1])
    values += 0.08 * seed + 0.01 * (np.random.rand(values.size) - 0.5)
    rho.vector().set_local(np.clip(values, 0.0, 1.0))
    rho.vector().apply("insert")
    return rho


def build_run_label():
    if OBJECTIVE_MODE == "poisson_ratio":
        ratio_tag = ("%.3f" % TARGET_POISSON_RATIO).replace("-", "m").replace(".", "p")
        return "poisson_ratio_nu_%s" % ratio_tag
    if OBJECTIVE_MODE == "orthotropic":
        return "orthotropic_%s" % ORTHO_PROFILE
    return OBJECTIVE_MODE


def print_run_header(run_output_dir):
    print("Objective mode: %s" % OBJECTIVE_MODE)
    if OBJECTIVE_MODE == "poisson_ratio":
        print("Target Poisson ratio: %.6f" % TARGET_POISSON_RATIO)
    elif OBJECTIVE_MODE == "orthotropic":
        print("Beam direction profile: %s" % ORTHO_PROFILE)
    elif OBJECTIVE_MODE == "shear_stiff":
        print("Targeting high effective shear stiffness (C66).")
    print("Output folder: %s" % run_output_dir)


def eval_cb(_j, m_rho):
    rt = filter_density(m_rho)
    rp = project_density(rt)
    assign(rho_viz, project(rp, Q))
    rho_file << rho_viz


rho = initialize_density()

output_dir = "output 1.5.2"
run_stamp = strftime("%Y%m%d_%H%M%S")
run_output_dir = os.path.join(output_dir, "%s_%s" % (build_run_label(), run_stamp))
os.makedirs(run_output_dir, exist_ok=True)

rho_file = File("%s/rho.pvd" % run_output_dir)
rho_bin_file = File("%s/rho_bin.pvd" % run_output_dir)
wa_file = File("%s/wA.pvd" % run_output_dir)
wb_file = File("%s/wB.pvd" % run_output_dir)
wy_file = File("%s/w_uniax_y.pvd" % run_output_dir)
rho_viz = Function(Q, name="rho")
stage_history = []

print_run_header(run_output_dir)


for stage_id, (penal_stage, beta_stage, max_iter_stage) in enumerate(continuation_stages, start=1):
    penal = penal_stage
    beta_proj = beta_stage
    print("Stage %d: penal=%.2f, beta=%.2f, iters=%d" % (stage_id, penal, beta_proj, max_iter_stage))

    rho_tilde = filter_density(rho)
    rho_phys = project_density(rho_tilde)

    C_stage, w_cases = compute_effective_stiffness_2d(rho_phys, "stage%d" % stage_id)
    objective, props_stage, objective_summary = build_objective_from_mode(C_stage, rho_phys, OBJECTIVE_MODE, ORTHO_PROFILE)

    vol_phys_stage = assemble(rho_phys * dx) / cell_volume
    print(
        "  Before opt: %s, C11=%.6f, C22=%.6f, C66=%.6f, vol_phys=%.6f"
        % (
            objective_summary,
            float(props_stage["C11"]),
            float(props_stage["C22"]),
            float(props_stage["C66"]),
            float(vol_phys_stage),
        )
    )

    m = Control(rho)
    Jhat = ReducedFunctional(objective, m, eval_cb_post=eval_cb)
    vol_form_upper = (rho - Constant(volfrac)) * dx
    vol_form_lower = (Constant(volfrac) - rho) * dx
    constraints = [
        UFLInequalityConstraint(vol_form_upper, m),
        UFLInequalityConstraint(vol_form_lower, m),
    ]
    problem_opt = MinimizationProblem(Jhat, bounds=(0.0, 1.0), constraints=constraints)
    solver_opt = IPOPTSolver(
        problem_opt,
        parameters={
            "maximum_iterations": max_iter_stage,
            "acceptable_tol": 1e-4,
            "tol": 1e-6,
            "print_level": 3,
        },
    )

    try:
        rho_star = solver_opt.solve()
        assign(rho, rho_star)
    except RuntimeError:
        print("  IPOPT stoppade tidigt i stage %d; fortsatter med senaste rho." % stage_id)

    rho_tilde = filter_density(rho)
    rho_phys = project_density(rho_tilde)
    assign(rho_viz, project(rho_phys, Q))
    rho_file << rho_viz
    wa_file << w_cases["11"]
    wb_file << w_cases["22"]

    C_post_stage, _ = compute_effective_stiffness_2d(rho_phys, "stage%d_post" % stage_id)
    objective_post, props_post, _ = build_objective_from_mode(C_post_stage, rho_phys, OBJECTIVE_MODE, ORTHO_PROFILE)
    stage_history.append({
        "stage": stage_id,
        "objective": float(objective_post),
        "nu_obj": float(effective_poisson_response(props_post["C11"], props_post["C12"], props_post["C21"], props_post["C22"])),
        "vol": float(assemble(rho_phys * dx) / cell_volume),
        "gray": float(compute_gray_density_metric(rho_phys)),
        "C11": float(props_post["C11"]),
        "C22": float(props_post["C22"]),
        "C66": float(props_post["C66"]),
    })

# Final small-strain report
rho_tilde = filter_density(rho)
rho_phys = project_density(rho_tilde)
C_final, w_final_cases = compute_effective_stiffness_2d(rho_phys, "final")
_, props_final, objective_summary_final = build_objective_from_mode(C_final, rho_phys, OBJECTIVE_MODE, ORTHO_PROFILE)
vol_final = assemble(rho_phys * dx) / cell_volume
gray_density_final = compute_gray_density_metric(rho_phys)
nu_obj_final = effective_poisson_response(props_final["C11"], props_final["C12"], props_final["C21"], props_final["C22"])

assign(rho_viz, project(rho_phys, Q))
rho_file << rho_viz

rho_bin = Function(Q, name="rho_bin")
rho_bin_expr = conditional(gt(rho_phys, Constant(bin_threshold)), Constant(1.0), Constant(0.0))
assign(rho_bin, project(rho_bin_expr, Q))
rho_bin_file << rho_bin

wa_file << w_final_cases["11"]
wb_file << w_final_cases["22"]

# Large-strain verification only. This does not affect the optimization.
z_final, l1_final_expr = solve_uniaxial_y_mixed(rho_phys, lambda2_max, "uy_final")
l1_final = assemble(l1_final_expr * dx) / cell_volume
nu_large = -(l1_final - 1.0) / (lambda2_max - 1.0)
wy_expr, _ = split(z_final)
wy = project(wy_expr, V)
wy.rename("w_uniax_y", "")
wy_file << wy

print(
    "Final small-strain: %s, volume=%.6f, nu_eff=%.6f, C11=%.6f, C22=%.6f, C66=%.6f"
    % (
        objective_summary_final,
        float(vol_final),
        float(nu_obj_final),
        float(props_final["C11"]),
        float(props_final["C22"]),
        float(props_final["C66"]),
    )
)
print_effective_stiffness_matrix(C_final)
print("Final objective-based Poisson response = %.6f" % float(nu_obj_final))
print("Final volume fraction = %.6f" % float(vol_final))
print("Remaining gray density before thresholding = %.6e" % float(gray_density_final))
print_convergence_history(stage_history)
print("Large-strain check @lambda2=%.2f: nu_eff=%.6f" % (lambda2_max, float(nu_large)))
print("Done. Open %s/rho_bin.pvd in ParaView." % run_output_dir)

"""
Vortex shedding behind a circular cylinder: OpenFOAM + Python
==============================================================
2D laminar flow past a cylinder at Re = 100 and Re = 200, fully scripted:

  1. Structured O-grid mesh written as a blockMeshDict (12 blocks, graded to the wall)
  2. Case setup: boundary conditions, physics and numerics
  3. Transient incompressible solver (foamRun, incompressibleFluid, PIMPLE)
  4. Drag and lift coefficients every time step
  5. Strouhal number from the lift signal (zero crossings and FFT)
  6. Validation against published experiments and simulations
  7. Mesh independence study (three meshes)
  8. Time-averaged flow: pressure coefficient on the wall and recirculation length
  9. Vorticity snapshots and an animation of the von Karman vortex street

Run it inside WSL/Linux with OpenFOAM 12 loaded:   python3 cylinder_vortex_shedding.py
Non-dimensional setup: D = 1, U = 1, so Re = 1/nu and time is in units of D/U.
"""
import os
import re
import glob
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import animation

# ============================ SETTINGS ============================
FLAVOR = "openfoam.org"     # "openfoam.org" (OpenFOAM 12) or "esi" (openfoam.com)
D, U_INF = 1.0, 1.0
SPAN = 0.1                  # depth of the 2D slab (only used for the reference area)
X_UP, X_DOWN, Y_SIDE = 10.0, 30.0, 10.0   # domain: 10 D upstream, 30 D downstream, 20 D wide
END_TIME = 150.0            # long enough for shedding to become fully periodic
AVG_START = 75.0            # statistics are taken from here to END_TIME
ANIM_TIME = 12.0            # extra time for the animation (about two shedding periods)
ANIM_DT = 0.5               # time between animation frames
MAX_CO = 0.8
CASES = [                   # name, Re, mesh scale
    ("Re100_coarse", 100, 0.5),
    ("Re100_medium", 100, 1.0),
    ("Re100_fine", 100, 1.5),
    ("Re200_medium", 200, 1.0),
]
MAIN = "Re100_medium"       # case used for the flow-field figures
RERUN = False               # False: reuse finished simulations and only redo the post-processing
N_JOBS = max(1, min(len(CASES), (os.cpu_count() or 2) - 1))
OUT = "results"
# ==================================================================

# Reference data (2D laminar flow, Re = 100 and 200)
REFERENCE = {
    100: dict(St=0.164, Cd=1.33, Cl=0.33,
              source="St: Williamson (1996) experiments; Cd, Cl: Park, Kwon & Choi (1998) simulations"),
    200: dict(St=0.196, Cd=1.34, Cl=0.68,
              source="2D simulations: Braza et al. (1986), Liu et al. (1998)"),
}


def williamson_st(re):
    """Williamson (1988) fit of experimental Strouhal number, laminar regime (49 < Re < 180)."""
    return -3.3265 / re + 0.1816 + 1.6e-4 * re


def run(cmd, case, log):
    with open(os.path.join(case, log), "w") as f:
        res = subprocess.run(cmd, cwd=case, stdout=f, stderr=subprocess.STDOUT)
    if res.returncode != 0:
        raise RuntimeError(f"'{' '.join(cmd)}' failed in {case}, see {case}/{log}")


def header(cls, obj, location=None):
    loc = f'    location    "{location}";\n' if location else ""
    return ("FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
            f"    class       {cls};\n{loc}    object      {obj};\n}}\n\n")


# ------------------------------------------------------------------
# 1. MESH
#    Inner O-grid: 4 blocks between the cylinder (r = 0.5) and a square
#    of half-size 2. Outer region: 8 blocks out to the far field.
# ------------------------------------------------------------------
def block_mesh_dict(scale):
    nc = int(round(40 * scale))          # cells per quarter of the cylinder
    nr = int(round(40 * scale))          # cells in the radial direction
    nup = int(round(40 * scale))         # upstream
    ndown = int(round(120 * scale))      # downstream (wake)
    nside = int(round(40 * scale))       # top and bottom
    g_r = 10.0                           # radial grading: last cell / first cell
    s = 2.0                              # half-size of the inner square
    xs = [-X_UP, -s, s, X_DOWN]
    ys = [-Y_SIDE, -s, s, Y_SIDE]
    r = D / 2
    c = r / np.sqrt(2)
    pts = [(x, y) for y in ys for x in xs]                 # 0..15 lattice
    pts += [(c, -c), (c, c), (-c, c), (-c, -c)]            # 16..19 on the cylinder
    n = len(pts)
    z0, z1 = -SPAN / 2, SPAN / 2

    def L(i, j):
        return j * 4 + i

    c0, c1, c2, c3 = 16, 17, 18, 19
    blocks = []

    def hexa(v, cells, grading):
        v = list(v)
        top = [k + n for k in v]
        blocks.append(f"    hex ({' '.join(map(str, v + top))}) ({cells[0]} {cells[1]} 1) "
                      f"simpleGrading ({grading[0]} {grading[1]} 1)")

    # inner O-grid (direction 1 = radial, direction 2 = around the cylinder)
    hexa((c0, L(2, 1), L(2, 2), c1), (nr, nc), (g_r, 1))
    hexa((c1, L(2, 2), L(1, 2), c2), (nr, nc), (g_r, 1))
    hexa((c2, L(1, 2), L(1, 1), c3), (nr, nc), (g_r, 1))
    hexa((c3, L(1, 1), L(2, 1), c0), (nr, nc), (g_r, 1))
    # outer blocks
    nx = [nup, nc, ndown]
    ny = [nside, nc, nside]
    gx = [0.2, 1, 5]
    gy = [0.2, 1, 5]
    for j in range(3):
        for i in range(3):
            if (i, j) == (1, 1):
                continue
            hexa((L(i, j), L(i + 1, j), L(i + 1, j + 1), L(i, j + 1)), (nx[i], ny[j]), (gx[i], gy[j]))

    verts = [f"    ({x:.6f} {y:.6f} {z0})" for x, y in pts] + [f"    ({x:.6f} {y:.6f} {z1})" for x, y in pts]
    arcs = []
    for a, b, mid in ((c0, c1, (r, 0)), (c1, c2, (0, r)), (c2, c3, (-r, 0)), (c3, c0, (0, -r))):
        for off, z in ((0, z0), (n, z1)):
            arcs.append(f"    arc {a + off} {b + off} ({mid[0]:.6f} {mid[1]:.6f} {z})")

    def face(a, b):
        return f"            ({a} {b} {b + n} {a + n})"

    cyl = [face(c0, c1), face(c1, c2), face(c2, c3), face(c3, c0)]
    inlet = [face(L(0, j + 1), L(0, j)) for j in range(3)]
    outlet = [face(L(3, j), L(3, j + 1)) for j in range(3)]
    bottom = [face(L(i, 0), L(i + 1, 0)) for i in range(3)]
    top = [face(L(i + 1, 3), L(i, 3)) for i in range(3)]

    def patch(name, ptype, faces):
        return f"    {name}\n    {{\n        type {ptype};\n        faces\n        (\n" + "\n".join(faces) + "\n        );\n    }\n"

    txt = header("dictionary", "blockMeshDict") + "convertToMeters 1;\n\nvertices\n(\n" + "\n".join(verts) + "\n);\n\n"
    txt += "blocks\n(\n" + "\n".join(blocks) + "\n);\n\nedges\n(\n" + "\n".join(arcs) + "\n);\n\nboundary\n(\n"
    txt += patch("inlet", "patch", inlet) + patch("outlet", "patch", outlet)
    txt += patch("top", "symmetryPlane", top) + patch("bottom", "symmetryPlane", bottom)
    txt += patch("cylinder", "wall", cyl)
    txt += "    frontAndBack\n    {\n        type empty;\n        faces ();\n    }\n);\n"
    return txt


# ------------------------------------------------------------------
# 2. CASE FILES
# ------------------------------------------------------------------
def write_case(case, re, scale, main):
    if os.path.exists(case):
        shutil.rmtree(case)
    for d in ("0", "constant", "system"):
        os.makedirs(os.path.join(case, d))
    nu = U_INF * D / re

    def w(path, text):
        with open(os.path.join(case, path), "w") as f:
            f.write(text)

    w("system/blockMeshDict", block_mesh_dict(scale))

    # initial and boundary conditions. A small cross-flow velocity in the
    # initial field breaks the symmetry so shedding starts early.
    w("0/U", header("volVectorField", "U", "0") + f"""dimensions      [0 1 -1 0 0 0 0];
internalField   uniform ({U_INF} 0.05 0);
boundaryField
{{
    inlet        {{ type fixedValue; value uniform ({U_INF} 0 0); }}
    outlet       {{ type inletOutlet; inletValue uniform (0 0 0); value uniform ({U_INF} 0 0); }}
    cylinder     {{ type noSlip; }}
    top          {{ type symmetryPlane; }}
    bottom       {{ type symmetryPlane; }}
    frontAndBack {{ type empty; }}
}}
""")
    w("0/p", header("volScalarField", "p", "0") + """dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    inlet        { type zeroGradient; }
    outlet       { type fixedValue; value uniform 0; }
    cylinder     { type zeroGradient; }
    top          { type symmetryPlane; }
    bottom       { type symmetryPlane; }
    frontAndBack { type empty; }
}
""")

    if FLAVOR == "openfoam.org":
        w("constant/physicalProperties", header("dictionary", "physicalProperties", "constant")
          + f"viscosityModel  constant;\n\nnu              [0 2 -1 0 0 0 0] {nu};\n")
        w("constant/momentumTransport", header("dictionary", "momentumTransport", "constant")
          + "simulationType  laminar;\n")
        app = "application     foamRun;\n\nsolver          incompressibleFluid;\n"
        forces = (f"    #includeFunc forceCoeffsIncompressible(patches = (cylinder), magUInf = {U_INF}, "
                  f"lRef = {D}, Aref = {D * SPAN}, CofR = (0 0 0), liftDir = (0 1 0), "
                  "dragDir = (1 0 0), pitchAxis = (0 0 1))\n")
    else:
        w("constant/transportProperties", header("dictionary", "transportProperties", "constant")
          + f"transportModel  Newtonian;\n\nnu              {nu};\n")
        w("constant/turbulenceProperties", header("dictionary", "turbulenceProperties", "constant")
          + "simulationType  laminar;\n")
        app = "application     pimpleFoam;\n"
        forces = f"""    forceCoeffs
    {{
        type            forceCoeffs;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   1;
        patches         (cylinder);
        rho             rhoInf;
        rhoInf          1;
        magUInf         {U_INF};
        lRef            {D};
        Aref            {D * SPAN};
        CofR            (0 0 0);
        liftDir         (0 1 0);
        dragDir         (1 0 0);
        pitchAxis       (0 0 1);
    }}
"""
    extra = ""
    if main:
        extra = f"""    #includeFunc vorticity
    fieldAverage1
    {{
        type            fieldAverage;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
        timeStart       {AVG_START};
        fields
        (
            U {{ mean on; prime2Mean off; base time; }}
            p {{ mean on; prime2Mean off; base time; }}
        );
    }}
"""
    w("system/controlDict", header("dictionary", "controlDict", "system") + app + f"""
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         {END_TIME};
deltaT          0.001;
writeControl    adjustableRunTime;
writeInterval   {END_TIME / 3};
purgeWrite      0;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   8;
runTimeModifiable true;
adjustTimeStep  yes;
maxCo           {MAX_CO};

functions
{{
{forces}{extra}}}
""")
    w("system/fvSchemes", header("dictionary", "fvSchemes", "system") + """ddtSchemes      { default backward; }
gradSchemes     { default Gauss linear; }
divSchemes
{
    default         Gauss linear;
    div(phi,U)      Gauss linearUpwind grad(U);
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
""")
    w("system/fvSolution", header("dictionary", "fvSolution", "system") + """solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-07;
        relTol          0.01;
    }
    pFinal
    {
        $p;
        relTol          0;
    }
    "(U|UFinal)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-08;
        relTol          0;
    }
}
PIMPLE
{
    nOuterCorrectors         1;
    nCorrectors              2;
    nNonOrthogonalCorrectors 0;
}
""")


def solver_cmd():
    return ["foamRun"] if FLAVOR == "openfoam.org" else ["pimpleFoam"]


def post_cmd():
    return ["foamPostProcess"] if FLAVOR == "openfoam.org" else ["postProcess"]


def set_control(case, end_time, write_interval):
    path = os.path.join(case, "system/controlDict")
    with open(path) as f:
        txt = f.read()
    txt = re.sub(r"\nendTime\s+[^;]+;", f"\nendTime         {end_time};", txt)
    txt = re.sub(r"\nwriteInterval\s+[^;]+;", f"\nwriteInterval   {write_interval};", txt)
    with open(path, "w") as f:
        f.write(txt)


def finished(case, log):
    path = os.path.join(case, log)
    if not os.path.exists(path):
        return False
    with open(path) as f:
        return f.read().rstrip().endswith("End")


def run_case(name, re_num, scale):
    main = name == MAIN
    done = finished(name, "log.solver") and (not main or os.path.exists(os.path.join(name, "0", "C")))
    if done and not RERUN:
        print(f"  {name}: results found, skipping the simulation (set RERUN = True to run it again)")
        return name
    write_case(name, re_num, scale, main)
    run(["blockMesh"], name, "log.blockMesh")
    run(["checkMesh"], name, "log.checkMesh")
    run(solver_cmd(), name, "log.solver")
    if main:   # extra run with frequent output, for the animation
        set_control(name, END_TIME + ANIM_TIME, ANIM_DT)
        run(solver_cmd(), name, "log.solver_anim")
        run(post_cmd() + ["-func", "writeCellCentres", "-time", "0"], name, "log.cellCentres")
    return name


# ------------------------------------------------------------------
# 3. READ RESULTS
# ------------------------------------------------------------------
def read_force_coeffs(case):
    """Cd and Cl histories, joined across restarts (one sub-folder per start time)."""
    files = glob.glob(os.path.join(case, "postProcessing", "**", "forceCoeffs*.dat"), recursive=True)
    data = []
    for fname in files:
        cols = None
        with open(fname) as f:
            for line in f:
                if line.startswith("#"):
                    toks = line[1:].split()
                    if toks and toks[0] == "Time":
                        cols = toks
                    continue
                vals = line.split()
                if cols and len(vals) == len(cols):
                    data.append([float(vals[0]), float(vals[cols.index("Cd")]), float(vals[cols.index("Cl")])])
    data = np.array(sorted(data))
    _, keep = np.unique(data[:, 0], return_index=True)
    data = data[keep]
    return data[:, 0], data[:, 1], data[:, 2]


def mesh_cells(case):
    with open(os.path.join(case, "log.checkMesh")) as f:
        return int(re.search(r"cells:\s+(\d+)", f.read()).group(1))


def shedding_stats(t, cd, cl):
    """Strouhal number (zero crossings and FFT), mean drag and lift amplitude."""
    m = t >= AVG_START
    t, cd, cl = t[m], cd[m], cl[m]
    # zero up-crossings of the lift signal, refined by linear interpolation
    i = np.where((cl[:-1] < 0) & (cl[1:] >= 0))[0]
    tc = t[i] - cl[i] * (t[i + 1] - t[i]) / (cl[i + 1] - cl[i])
    period = np.mean(np.diff(tc))
    st_zc = D / (U_INF * period)
    # FFT on a uniform time grid over a whole number of periods, zero-padded
    tu = np.linspace(tc[0], tc[-1], 4096, endpoint=False)
    clu = np.interp(tu, t, cl)
    nfft = 2 ** 18
    spec = np.abs(np.fft.rfft((clu - clu.mean()) * np.hanning(len(clu)), nfft))
    freq = np.fft.rfftfreq(nfft, tu[1] - tu[0])
    st_fft = freq[np.argmax(spec)] * D / U_INF
    w = (t >= tc[0]) & (t <= tc[-1])
    return dict(St=st_zc, St_fft=st_fft, Cd=np.mean(cd[w]), Cd_amp=(cd[w].max() - cd[w].min()) / 2,
                Cl=(cl[w].max() - cl[w].min()) / 2, periods=len(tc) - 1,
                freq=freq * D / U_INF, spec=spec / spec.max())


def read_field(path, kind):
    """Internal field and boundary values of an ASCII OpenFOAM field."""
    with open(path) as f:
        txt = f.read()

    def parse_list(chunk):
        n = int(re.match(r"\s*(\d+)\s*\(", chunk).group(1))
        body = chunk[chunk.index("(") + 1:]
        if kind == "vector":
            vals = re.findall(r"\(([^()]+)\)", body)[:n]
            return np.array([[float(v) for v in s.split()] for s in vals])
        return np.array(body.split(")")[0].split(), dtype=float)[:n]

    m = re.search(r"internalField\s+nonuniform\s+List<\w+>", txt)
    internal = parse_list(txt[m.end():])
    patches = {}
    for name in ("cylinder", "inlet"):
        pm = re.search(name + r"\s*\{[^}]*?value\s+nonuniform\s+List<\w+>", txt, re.S)
        if pm:
            patches[name] = parse_list(txt[pm.end():])
    return internal, patches


def latest_time(case):
    times = [d for d in os.listdir(case) if re.fullmatch(r"[0-9.eE+-]+", d) and d != "0"]
    return sorted(times, key=float)


# ------------------------------------------------------------------
# 4. FLOW FIELD PLOTS
# ------------------------------------------------------------------
def triangulation(xy):
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1])
    p = xy[tri.triangles]
    cen = p.mean(axis=1)
    edge = np.max(np.linalg.norm(p - np.roll(p, 1, axis=1), axis=2), axis=1)
    tri.set_mask((np.hypot(cen[:, 0], cen[:, 1]) < D / 2 * 1.02) | (edge > 0.6))
    return tri


def draw_cylinder(ax):
    ax.add_patch(plt.Circle((0, 0), D / 2, color="0.25", zorder=5))


# ==================================================================
def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"Running {len(CASES)} OpenFOAM cases, {N_JOBS} at a time. "
          "This takes a while (see each case folder for the logs)...")
    with ThreadPoolExecutor(max_workers=N_JOBS) as pool:
        list(pool.map(lambda c: run_case(*c), CASES))

    report = []

    def log(text=""):
        print(text)
        report.append(text)

    # ---------------- Forces and Strouhal number ----------------
    results = {}
    for name, re_num, scale in CASES:
        t, cd, cl = read_force_coeffs(name)
        t_end = t <= END_TIME
        s = shedding_stats(t[t_end], cd[t_end], cl[t_end])
        s.update(name=name, Re=re_num, cells=mesh_cells(name), t=t, cd=cd, cl=cl)
        results[name] = s

    log("FLOW PAST A CIRCULAR CYLINDER, 2D laminar, OpenFOAM")
    log(f"Domain: {X_UP:.0f} D upstream, {X_DOWN:.0f} D downstream, {2 * Y_SIDE:.0f} D wide "
        f"(blockage {D / (2 * Y_SIDE) * 100:.0f}%) | statistics from t = {AVG_START:.0f} to {END_TIME:.0f} D/U\n")
    log(f"{'Case':<14} {'Cells':>7} {'Periods':>7} {'St':>7} {'St FFT':>7} {'Cd mean':>8} {'Cl amp':>7} {'Cd amp':>7}")
    for s in results.values():
        log(f"{s['name']:<14} {s['cells']:>7} {s['periods']:>7} {s['St']:>7.4f} {s['St_fft']:>7.4f} "
            f"{s['Cd']:>8.4f} {s['Cl']:>7.4f} {s['Cd_amp']:>7.4f}")

    log("\nVALIDATION")
    for re_num in (100, 200):
        s = results["Re100_fine" if re_num == 100 else "Re200_medium"]
        ref = REFERENCE[re_num]
        log(f"  Re = {re_num} ({s['name']}):")
        log(f"    St      = {s['St']:.4f}  vs {ref['St']:.3f}  ({(s['St'] - ref['St']) / ref['St'] * 100:+.1f}%)")
        log(f"    Cd mean = {s['Cd']:.3f}   vs {ref['Cd']:.2f}   ({(s['Cd'] - ref['Cd']) / ref['Cd'] * 100:+.1f}%)")
        log(f"    Cl amp  = {s['Cl']:.3f}   vs {ref['Cl']:.2f}   ({(s['Cl'] - ref['Cl']) / ref['Cl'] * 100:+.1f}%)")
        log(f"    Reference: {ref['source']}")
    log(f"  Williamson's experimental fit gives St = {williamson_st(100):.4f} at Re = 100")

    mesh_names = ["Re100_coarse", "Re100_medium", "Re100_fine"]
    log("\nMESH INDEPENDENCE (Re = 100)")
    f_ = results["Re100_fine"]
    for nm in mesh_names[:2]:
        s = results[nm]
        log(f"  {nm:<14} vs fine: St {(s['St'] - f_['St']) / f_['St'] * 100:+.2f}%, "
            f"Cd {(s['Cd'] - f_['Cd']) / f_['Cd'] * 100:+.2f}%, Cl {(s['Cl'] - f_['Cl']) / f_['Cl'] * 100:+.2f}%")

    # ---------------- Figure: force history ----------------
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for nm, col in (("Re100_medium", "tab:blue"), ("Re200_medium", "tab:red")):
        s = results[nm]
        m = s["t"] <= END_TIME
        axes[0].plot(s["t"][m], s["cd"][m], color=col, lw=0.8, label=f"Re = {s['Re']}")
        axes[1].plot(s["t"][m], s["cl"][m], color=col, lw=0.8, label=f"Re = {s['Re']}")
    for ax in axes:
        ax.axvspan(AVG_START, END_TIME, color="0.9", zorder=0)
        ax.grid(alpha=0.3)
    axes[0].text(AVG_START + 1, 1.0, "statistics window", fontsize=8, color="0.4")
    axes[0].set_ylabel("Drag coefficient Cd")
    axes[0].set_ylim(0.9, 2.0)
    axes[1].set_ylim(-1.0, 1.0)
    axes[1].set_ylabel("Lift coefficient Cl")
    axes[1].set_xlabel("Time t U / D")
    axes[0].legend(loc="upper right")
    fig.suptitle("Force history: start-up, onset of shedding and periodic regime", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/force_history.png", dpi=150)
    plt.close(fig)

    # ---------------- Figure: spectrum and zoom ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    s = results[MAIN]
    m = (s["t"] >= END_TIME - 3 / s["St"]) & (s["t"] <= END_TIME)
    ax = axes[0]
    ax.plot(s["t"][m], s["cd"][m], color="tab:blue", label="Cd")
    ax2 = ax.twinx()
    ax2.plot(s["t"][m], s["cl"][m], color="tab:orange", label="Cl")
    ax.set_xlabel("Time t U / D")
    ax.set_ylabel("Cd", color="tab:blue")
    ax2.set_ylabel("Cl", color="tab:orange")
    ax.set_title("Three shedding periods at Re = 100:\ndrag oscillates at twice the lift frequency")
    ax.grid(alpha=0.3)
    ax = axes[1]
    for nm, col in (("Re100_medium", "tab:blue"), ("Re200_medium", "tab:red")):
        r_ = results[nm]
        ax.plot(r_["freq"], r_["spec"], color=col, label=f"Re = {r_['Re']}: St = {r_['St']:.3f}")
        ax.axvline(REFERENCE[r_["Re"]]["St"], color=col, ls=":", lw=1)
    ax.set_xlim(0, 0.6)
    ax.set_xlabel("Strouhal number f D / U")
    ax.set_ylabel("Normalized lift spectrum")
    ax.set_title("Lift spectrum (dotted lines: reference St)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/spectrum.png", dpi=150)
    plt.close(fig)

    # ---------------- Figure: mesh study ----------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    cells = [results[nm]["cells"] for nm in mesh_names]
    for ax, key, lab in zip(axes, ("St", "Cd", "Cl"), ("Strouhal number", "Mean drag Cd", "Lift amplitude Cl")):
        ax.plot(cells, [results[nm][key] for nm in mesh_names], "o-", color="k")
        ax.axhline(REFERENCE[100][key], color="tab:green", ls="--", label="reference")
        ax.set_xscale("log")
        ax.set_xlabel("Number of cells")
        ax.set_title(lab)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend()
    fig.suptitle("Mesh independence at Re = 100", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/mesh_study.png", dpi=150)
    plt.close(fig)

    # ---------------- Flow fields of the main case ----------------
    C, Cb = read_field(os.path.join(MAIN, "0", "C"), "vector")
    xy = C[:, :2]
    view = (xy[:, 0] > -3) & (xy[:, 0] < 16) & (np.abs(xy[:, 1]) < 4.5)
    tri = triangulation(xy[view])
    times = latest_time(MAIN)
    anim_times = [tm for tm in times if float(tm) > END_TIME + 1e-6]

    def vort(tm):
        w, _ = read_field(os.path.join(MAIN, tm, "vorticity"), "vector")
        return w[view, 2]

    levels = np.linspace(-3, 3, 41)

    # vorticity snapshot
    fig, ax = plt.subplots(figsize=(12, 4))
    cf = ax.tricontourf(tri, np.clip(vort(anim_times[0]), -3, 3), levels=levels, cmap="RdBu_r", extend="both")
    draw_cylinder(ax)
    ax.set_aspect("equal")
    ax.set_xlim(-3, 16)
    ax.set_ylim(-4, 4)
    ax.set_xlabel("x / D")
    ax.set_ylabel("y / D")
    ax.set_title(f"Von Karman vortex street at Re = 100 (vorticity, t = {float(anim_times[0]):.1f} D/U)")
    fig.colorbar(cf, ax=ax, label="Vorticity ωz D / U", shrink=0.8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/vorticity.png", dpi=150)
    plt.close(fig)

    # animation
    fig, ax = plt.subplots(figsize=(10, 3.6))
    frames = [np.clip(vort(tm), -3, 3) for tm in anim_times]

    def frame(k):
        ax.cla()
        ax.tricontourf(tri, frames[k], levels=levels, cmap="RdBu_r", extend="both")
        draw_cylinder(ax)
        ax.set_aspect("equal")
        ax.set_xlim(-3, 16)
        ax.set_ylim(-4, 4)
        ax.set_axis_off()
        ax.set_title(f"Re = 100, vorticity, t = {float(anim_times[k]):.1f} D/U", fontsize=10)
        return []

    anim = animation.FuncAnimation(fig, frame, frames=len(frames))
    anim.save(f"{OUT}/vortex_street.gif", writer=animation.PillowWriter(fps=8))
    plt.close(fig)

    # time-averaged flow: Cp on the wall and recirculation length
    t_last = times[-1]
    p_mean, p_b = read_field(os.path.join(MAIN, t_last, "pMean"), "scalar")
    u_mean, _ = read_field(os.path.join(MAIN, t_last, "UMean"), "vector")
    fc = Cb["cylinder"][:, :2]
    theta = np.degrees(np.arctan2(fc[:, 1], -fc[:, 0]))      # 0 deg = front stagnation point
    p_inf = np.mean(p_b["inlet"])
    cp = (p_b["cylinder"] - p_inf) / (0.5 * U_INF ** 2)
    upper = theta >= 0
    order = np.argsort(theta[upper])
    th, cpu = theta[upper][order], cp[upper][order]
    log("\nTIME-AVERAGED FLOW (Re = 100)")
    log(f"  Cp at the front stagnation point = {cpu[0]:.3f}, base pressure Cp(180 deg) = {cpu[-1]:.3f}")

    line = (np.abs(xy[:, 1]) < 0.06) & (xy[:, 0] > D / 2)
    xl = xy[line, 0]
    ux = u_mean[line, 0]
    o = np.argsort(xl)
    xl, ux = xl[o], ux[o]
    k = np.where((ux[:-1] < 0) & (ux[1:] >= 0))[0][0]
    x_r = xl[k] - ux[k] * (xl[k + 1] - xl[k]) / (ux[k + 1] - ux[k])
    lr = (x_r - D / 2) / D
    log(f"  Mean recirculation length behind the cylinder L_r = {lr:.2f} D")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.plot(th, cpu, color="k", lw=2, label="OpenFOAM, time average")
    tp = np.linspace(0, 180, 200)
    ax.plot(tp, 1 - 4 * np.sin(np.radians(tp)) ** 2, "--", color="gray", label="Potential flow (no viscosity)")
    ax.set_xlabel("Angle from the front stagnation point θ [deg]")
    ax.set_ylabel("Pressure coefficient Cp")
    ax.set_ylim(-3.2, 1.3)
    ax.set_xlim(0, 180)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Mean wall pressure: separation stops the pressure recovery")
    ax = axes[1]
    um = u_mean[view]
    speed = np.hypot(um[:, 0], um[:, 1])
    cf = ax.tricontourf(tri, speed, levels=30, cmap="viridis")
    ax.tricontour(tri, um[:, 0], levels=[0], colors="w", linewidths=1.2)
    draw_cylinder(ax)
    ax.set_aspect("equal")
    ax.set_xlim(-2, 6)
    ax.set_ylim(-2, 2)
    ax.set_title(f"Mean velocity (white line: Ux = 0)\nrecirculation bubble L_r = {lr:.2f} D", fontsize=10)
    fig.colorbar(cf, ax=ax, label="|U mean| / U", shrink=0.8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/mean_flow.png", dpi=150)
    plt.close(fig)

    # mesh picture
    fig, ax = plt.subplots(figsize=(7, 6))
    pts_near = xy[(np.abs(xy[:, 0]) < 2.5) & (np.abs(xy[:, 1]) < 2.5)]
    ax.plot(pts_near[:, 0], pts_near[:, 1], ".", ms=1.2, color="k")
    draw_cylinder(ax)
    ax.set_aspect("equal")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_title(f"O-grid around the cylinder (cell centres), {results[MAIN]['cells']} cells in total")
    fig.tight_layout()
    fig.savefig(f"{OUT}/mesh.png", dpi=150)
    plt.close(fig)

    with open(f"{OUT}/report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"\nFigures and report saved in the '{OUT}' folder.")


if __name__ == "__main__":
    main()

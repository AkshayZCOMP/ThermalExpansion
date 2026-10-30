# Abaqus shell-based thermoelastic laminate model with inertia relief
# One-step trial:
#   Step-1: thermal loading and structural response with inertia relief
#   No encastre or other displacement/rotation boundary conditions
#
# SI units (m, kg, s, N, Pa, W, J)

from abaqus import *
from abaqusConstants import *
import regionToolset
import mesh
import math
import os
import ast

# ==============================================================================
# 1. INPUT PARAMETERS
# ==============================================================================
MODEL_NAME = 'Model_shell_IR'
PART_NAME   = 'LaminateShell'
MAT_NAME    = 'Material-1'
JOB_NAME    = 'LaminateShellIRJob'
SUMMARY_CSV_NAME = 'ABaqusData.csv'

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

# Geometry [m]
LENGTH = 0.3
WIDTH  = 0.3
PLY_THICKNESS = 0.000125
LAYUP = [0,90,90,0]#[90,0,0,90][-45/45/45/-45/90/-45/45/45/-45]
MESH_SIZE = 0.005
# Thermal / initial conditions
INITIAL_TEMP = 25.0
T_bot_sink   = 100
T_top_sink   = 0.0
h_bot        = 1000.0
h_top        = 1000 
kz           = 0.5
TIME_PERIOD  = 10.0

# Use one prescribed thermal field for every laminate case.  The midplane
# temperature stays fixed while top/bottom temperatures follow from thickness.
FIXED_MIDPLANE_TEMPERATURE = 50.0
FIXED_THERMAL_GRADIENT = -53333.3  # C/m, where z=0 is the laminate midplane

if os.environ.get('ABAQUS_LAYUP'):
    LAYUP = ast.literal_eval(os.environ['ABAQUS_LAYUP'])

if os.environ.get('ABAQUS_JOB_NAME'):
    JOB_NAME = os.environ['ABAQUS_JOB_NAME']

NPLIES = len(LAYUP)
TOTAL_THICKNESS = NPLIES * PLY_THICKNESS

# Fixed thermal field shared with Computational Real/FREEFREE.py.
T_mid = FIXED_MIDPLANE_TEMPERATURE
beta = FIXED_THERMAL_GRADIENT
T_top_surf = T_mid + beta * TOTAL_THICKNESS / 2.0
T_bot_surf = T_mid - beta * TOTAL_THICKNESS / 2.0
# ==============================================================================
# 2. MODEL
# ==============================================================================
if MODEL_NAME in mdb.models.keys():
    del mdb.models[MODEL_NAME]
model = mdb.Model(name=MODEL_NAME)
model.setValues(absoluteZero=-273.15)

# ==============================================================================
# 3. GEOMETRY (SHELL MID-SURFACE)
# ==============================================================================
s = model.ConstrainedSketch(name='profile', sheetSize=1.0)
s.rectangle((0.0, 0.0), (LENGTH, WIDTH))
part = model.Part(name=PART_NAME, dimensionality=THREE_D, type=DEFORMABLE_BODY)
part.BaseShell(sketch=s)
del model.sketches['profile']

# ==============================================================================
# 4. MATERIAL
# ==============================================================================
mat = model.Material(name=MAT_NAME)
mat.Density(table=((1580.0,),))
mat.Elastic(
    type=ENGINEERING_CONSTANTS,
    table=((130e9, 10e9, 10e9, 0.3, 0.3, 0.3, 5e9, 5e9, 5e9),)
)
mat.Expansion(type=ORTHOTROPIC, table=((-5e-7, 3e-5, 3e-5),))

# ==============================================================================
# 5. COMPOSITE LAYUP
# ==============================================================================
layup = part.CompositeLayup(
    name='Layup-1',
    description='Composite shell layup',
    elementType=SHELL,
    symmetric=False,
    thicknessAssignment=FROM_SECTION
)

layup.Section(
    preIntegrate=OFF,
    integrationRule=SIMPSON,
    thicknessType=UNIFORM,
    poissonDefinition=DEFAULT,
    temperature=GRADIENT,
    useDensity=ON
)

layup.ReferenceOrientation(
    orientationType=GLOBAL,
    localCsys=None,
    fieldName='',
    additionalRotationType=ROTATION_NONE,
    angle=0.0,
    axis=AXIS_3
)

ply_region = regionToolset.Region(faces=part.faces)
for i, ang in enumerate(LAYUP):
    layup.CompositePly(
        suppressed=False,
        plyName='Ply-%d' % (i + 1),
        region=ply_region,
        material=MAT_NAME,
        thicknessType=SPECIFY_THICKNESS,
        thickness=PLY_THICKNESS,
        orientationType=SPECIFY_ORIENT,
        orientationValue=float(ang),
        additionalRotationType=ROTATION_NONE,
        axis=AXIS_3,
        numIntPoints=3
    )

# ==============================================================================
# 6. ASSEMBLY
# ==============================================================================
a = model.rootAssembly
a.DatumCsysByDefault(CARTESIAN)
inst = a.Instance(name=PART_NAME + '-1', part=part, dependent=ON)

whole_faces = regionToolset.Region(faces=inst.faces)

# ==============================================================================
# 7. ONE THERMAL/STRUCTURAL STATIC STEP
# ==============================================================================
# The temperature changes from INITIAL_TEMP in the Initial step to the
# prescribed through-thickness temperature gradient in this analysis step.
model.StaticStep(
    name='Step-1-Thermal',
    previous='Initial',
    timePeriod=TIME_PERIOD,
    nlgeom=ON,
    initialInc=0.001,
    minInc=0.0001,
    maxInc=10.0,
    maxNumInc=1000
)

# ==============================================================================
# 8. BOUNDARY CONDITIONS
# ==============================================================================
# No displacement/rotation boundary conditions.
# The shell is free, with inertia relief active in both static steps.

# ==============================================================================
# 9. INITIAL TEMPERATURE
# ==============================================================================
model.Temperature(
    name='InitialTemp',
    createStepName='Initial',
    region=whole_faces,
    distributionType=UNIFORM,
    crossSectionDistribution=CONSTANT_THROUGH_THICKNESS,
    magnitudes=(INITIAL_TEMP,)
)

# ==============================================================================
# 10. PRESCRIBED THERMAL GRADIENT IN STEP-1
# ==============================================================================
model.Temperature(
    name='ThermalGradient',
    createStepName='Step-1-Thermal',
    region=whole_faces,
    distributionType=UNIFORM,
    crossSectionDistribution=GRADIENTS_THROUGH_SHELL_CS,
    magnitudes=(T_mid, beta)
)

# ==============================================================================
# 11. INERTIA RELIEF
# ==============================================================================
model.InertiaRelief(
    name='IR-Step1',
    createStepName='Step-1-Thermal',
    u1=ON,
    u2=ON,
    u3=ON,
    ur1=ON,
    ur2=ON,
    ur3=ON
)



# ==============================================================================
# 12. MESH
# ==============================================================================
part.seedPart(size=MESH_SIZE, deviationFactor=0.1, minSizeFactor=0.1)
et1 = mesh.ElemType(elemCode=S8R, elemLibrary=STANDARD)
#et2 = mesh.ElemType(elemCode=S3, elemLibrary=STANDARD)
part.setElementType(regions=(part.faces,), elemTypes=(et1,))
part.generateMesh()
a.regenerate()

# ==============================================================================
# 13. OUTPUT
# ==============================================================================
try:
    if 'F-Output-1' in model.fieldOutputRequests.keys():
        model.fieldOutputRequests['F-Output-1'].setValues(
            variables=('S', 'E', 'LE', 'U', 'RF', 'COORD', 'NT')
        )
    else:
        model.FieldOutputRequest(
            name='F-Output-1',
            createStepName='Step-1-Thermal',
            variables=('S', 'E', 'LE', 'U', 'RF', 'COORD', 'NT')
        )
except AttributeError:
    try:
        model.FieldOutputRequest(
            name='F-Output-1',
            createStepName='Step-1-Thermal',
            variables=('S', 'E', 'LE', 'U', 'RF', 'COORD', 'NT')
        )
    except AttributeError:
        print('Warning: field output request API was not available; using Abaqus defaults.')

# ==============================================================================
# 14. JOB
# ==============================================================================
mdb.Job(
    name=JOB_NAME,
    model=MODEL_NAME,
    type=ANALYSIS,
    # This avoids an Abaqus launcher CPU-discovery failure on this machine.
    numCpus=1,
    numDomains=1
)

mdb.jobs[JOB_NAME].submit()
mdb.jobs[JOB_NAME].waitForCompletion()

print('Finished one-step shell thermal analysis with inertia relief.')
# ==============================================================================
# 15. POSTPROCESSING: extract e0 and K from shell section-point strains
# ==============================================================================
from odbAccess import openOdb
from collections import defaultdict
import math

ODB_NAME = JOB_NAME + '.odb'
STEP_NAME = 'Step-1-Thermal'

odb = openOdb(ODB_NAME, readOnly=True)

# --- Find the instance automatically ---
inst_name = None
for k in odb.rootAssembly.instances.keys():
    if PART_NAME.upper() in k.upper():
        inst_name = k
        break

if inst_name is None:
    raise RuntimeError('Could not find an instance matching part name: %s' % PART_NAME)

print('Using ODB instance:', inst_name)

# --- Get final frame of the structural step ---
if STEP_NAME not in odb.steps.keys():
    raise RuntimeError('Step not found in ODB: %s' % STEP_NAME)

step = odb.steps[STEP_NAME]
frame = step.frames[-1]

# LE is usually the available strain output for this shell/nlgeom case.
if 'LE' in frame.fieldOutputs.keys():
    Efield = frame.fieldOutputs['LE']
elif 'E' in frame.fieldOutputs.keys():
    Efield = frame.fieldOutputs['E']
else:
    raise RuntimeError('Neither LE nor E strain output was found in the final frame.')

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------
def rotate_engineering_strain_2d(e11_l, e22_l, g12_l, theta_deg):
    """
    Transform engineering strain [e11, e22, gamma12] from ply axes to laminate axes.

    Important:
      - g12_l is engineering shear gamma12, not tensor shear e12.
      - Returned gxy is engineering shear gamma_xy.
    """
    th = math.radians(theta_deg)
    m = math.cos(th)
    n = math.sin(th)

    ex = m*m*e11_l + n*n*e22_l - m*n*g12_l
    ey = n*n*e11_l + m*m*e22_l + m*n*g12_l
    gxy = 2.0*m*n*(e11_l - e22_l) + (m*m - n*n)*g12_l

    return ex, ey, gxy


def fit_line(z_vals, y_vals):
    """Return (intercept, slope) for y = intercept + slope*z."""
    if len(z_vals) != len(y_vals):
        raise ValueError('z/y length mismatch')

    n = float(len(z_vals))
    if n <= 0.0:
        raise RuntimeError('Cannot fit line: empty data.')

    z_bar = sum(z_vals) / n
    y_bar = sum(y_vals) / n

    denom = 0.0
    numer = 0.0
    for z, y in zip(z_vals, y_vals):
        dz = z - z_bar
        dy = y - y_bar
        denom += dz * dz
        numer += dz * dy

    if abs(denom) < 1.0e-30:
        raise RuntimeError('Cannot fit line: zero z variance.')

    slope = numer / denom
    intercept = y_bar - slope * z_bar
    return intercept, slope


# --- Group strain values by element and section point ---
elem_data = defaultdict(list)

for v in Efield.values:
    if v.instance.name != inst_name:
        continue
    if v.sectionPoint is None:
        continue

    sp_num = v.sectionPoint.number
    elem_data[v.elementLabel].append((sp_num, v.data))

# --- Accumulate plate averages ---
sum_e0_11 = sum_e0_22 = sum_e0_12 = 0.0
sum_K_11 = sum_K_22 = sum_K_12 = 0.0
n_elem = 0

N_IP_PER_PLY = 3

# Abaqus section point order must match this assumption.
# Use (1.0, 0.5, 0.0) instead if your section descriptions show top/mid/bottom.
SECTION_Z_FRACTIONS = (0.0, 0.5, 1.0)

# ODB field-output LE12 is already Abaqus's engineering shear component.
# The laminate transformation below expects engineering gamma12, so do not
# apply another factor of two. (Abaqus's principal-strain expression divides
# LE12 by two when it needs the tensor shear component.)
ABAQUS_LE12_TO_GAMMA12 = 1.0

for elem_label in sorted(elem_data.keys()):
    rows = elem_data[elem_label]
    rows.sort(key=lambda x: x[0])

    z_list = []
    ex_list = []
    ey_list = []
    gxy_list = []

    for sp_num, data in rows:
        # Map section point -> ply index, assuming 3 integration points per ply.
        ply_index = min((sp_num - 1) // N_IP_PER_PLY, len(LAYUP) - 1)
        ip_in_ply = (sp_num - 1) % N_IP_PER_PLY
        ply_theta = float(LAYUP[ply_index])

        # Abaqus LE components:
        #   data[0] = local 11 normal strain
        #   data[1] = local 22 normal strain
        #   data[3] = local 12 shear component
        e11_l = data[0]
        e22_l = data[1]
        g12_l = ABAQUS_LE12_TO_GAMMA12 * data[3]

        # Transform from ply material axes to laminate/global axes.
        e11_g, e22_g, g12_g = rotate_engineering_strain_2d(
            e11_l, e22_l, g12_l, ply_theta
        )

        # Physical z-location of section point.
        z = (
            -0.5 * TOTAL_THICKNESS
            + ply_index * PLY_THICKNESS
            + SECTION_Z_FRACTIONS[ip_in_ply] * PLY_THICKNESS
        )

        z_list.append(z)
        ex_list.append(e11_g)
        ey_list.append(e22_g)
        gxy_list.append(g12_g)

    # Fit strain(z) = e0 + z*K
    e0_11, K_11 = fit_line(z_list, ex_list)
    e0_22, K_22 = fit_line(z_list, ey_list)
    e0_12, K_12 = fit_line(z_list, gxy_list)

    sum_e0_11 += e0_11
    sum_e0_22 += e0_22
    sum_e0_12 += e0_12
    sum_K_11 += K_11
    sum_K_22 += K_22
    sum_K_12 += K_12
    n_elem += 1

if n_elem > 0:
    avg_e0_11 = sum_e0_11 / n_elem
    avg_e0_22 = sum_e0_22 / n_elem
    avg_e0_12 = sum_e0_12 / n_elem
    avg_K_11 = sum_K_11 / n_elem
    avg_K_22 = sum_K_22 / n_elem
    avg_K_12 = sum_K_12 / n_elem

    print('--- Plate-averaged generalized strains (laminate coordinates) ---')
    print('e0_11 = % .6e' % avg_e0_11)
    print('e0_22 = % .6e' % avg_e0_22)
    print('e0_12 = % .6e' % avg_e0_12)
    print('K_11  = % .6e' % avg_K_11)
    print('K_22  = % .6e' % avg_K_22)
    print('K_12  = % .6e' % avg_K_12)

    summary_csv_path = os.environ.get(
        'ABAQUS_SUMMARY_CSV', os.path.join(SCRIPT_DIR, 'LayerEffect.csv')
    )
    summary_dir = os.path.dirname(summary_csv_path)
    if summary_dir and not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    write_header = (
        not os.path.exists(summary_csv_path) or
        os.path.getsize(summary_csv_path) == 0
    )
    layup_text = '[' + '/'.join(['%g' % float(ang) for ang in LAYUP]) + ']'

    with open(summary_csv_path, 'a') as summary:
        if write_header:
            summary.write(
                'Layup,NumberOfPlies,TopSinkTemperature,BottomSinkTemperature,'
                'TopTemperature,BottomTemperature,'
                'e0_11,e0_22,e0_12,K_11,K_22,K_12\n'
            )
        summary.write(
            '%s,%d,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e\n' %
            (
                layup_text,
                NPLIES,
                T_top_sink,
                T_bot_sink,
                T_top_surf,
                T_bot_surf,
                avg_e0_11,
                avg_e0_22,
                avg_e0_12,
                avg_K_11,
                avg_K_22,
                avg_K_12
            )
        )
    print('Appended run summary to %s' % summary_csv_path)

#     # The sweep runner supplies a project-local result file.  Keeping this
#     # configurable also lets a one-off Abaqus run write beside this script.
#     summary_csv_path = r'C:\Users\prz\Documents\HeatXfireDIC\HeatXferDIC\LayupSweep\LayerEffect.csv'
#     summary_dir = os.path.dirname(summary_csv_path)
#     if summary_dir and not os.path.exists(summary_dir):
#         os.makedirs(summary_dir)
#     write_header = (
#         not os.path.exists(summary_csv_path) or
#         os.path.getsize(summary_csv_path) == 0
#     )
#     layup_text = '[' + '/'.join(['%g' % float(ang) for ang in LAYUP]) + ']'

#     summary = open(summary_csv_path, 'a')
#     if write_header:
#         summary.write(
#             'Layup,NumberOfPlies,TopSinkTemperature,BottomSinkTemperature,'
#             'TopTemperature,BottomTemperature,'
#             'e0_11,e0_22,e0_12,K_11,K_22,K_12\n'
#         )
#     summary.write(
#         '%s,%d,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e\n' %
#         (
#             layup_text,
#             NPLIES,
#             T_top_sink,
#             T_bot_sink,
#             T_top_surf,
#             T_bot_surf,
#             avg_e0_11,
#             avg_e0_22,
#             avg_e0_12,
#             avg_K_11,
#             avg_K_22,
#             avg_K_12
#         )
#     )
#     summary.close()
#     print('Appended run summary to %s' % summary_csv_path)

# else:
#     print('No shell section-point strain data found.')


odb.close()

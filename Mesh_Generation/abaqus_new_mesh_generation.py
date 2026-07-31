from abaqus import *
from abaqusConstants import *
# -*- coding: mbcs -*-
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
import os
import shutil

# Define path to save
wk_dir = 'C:\\path\\to\\save\\'
# Following results obtained from running "Dimple_thickness_calculation.ipynb"
root = 'HT_250818'
thickness = [['A5', 0.568], ['B5', 0.544], ['C5', 0.57], ['D5', 0.462], ['E5', 0.471], ['A4', 0.495], ['B4', 0.503], ['C4', 0.523], ['D4', 0.5], ['E4', 0.494], ['A3', 0.514], ['B3', 0.489], ['C3', 0.481], ['D3', 0.471], ['E3', 0.502], ['A2', 0.504], ['B2', 0.464], ['C2', 0.48], ['D2', 0.489], ['E2', 0.47], ['A1', 0.483], ['B1', 0.467], ['C1', 0.488], ['D1', 0.472], ['E1', 0.496]]

# Function generates Abaqus axisymmetrical dimple geometry with different dimple thicknesses. 
# Code generates geometry, creates sample material, meshes part, and automatically 
# creates needed node sets and surfaces.
# Creates mesh in ElemType CAX4R, modified to CPS4 after generation for JAX-FEM to process
def generate_models(root, input_list, main_dir=wk_dir):
    Mdb()
    for l in input_list:
        name, t = l
        model = 'Model_%s' % name
        m = mdb.Model(name=model)
        # t = thickness[i]
        m.ConstrainedSketch(name='__profile__', sheetSize=100.0)
        m.sketches['__profile__'].sketchOptions.setValues(viewStyle=AXISYM)
        m.sketches['__profile__'].ConstructionLine(point1=(0.0, -100.0), point2=(0.0, 100.0))
        m.sketches['__profile__'].FixedConstraint(entity=m.sketches['__profile__'].geometry[2])
        center_y = 3 - 1.5 - t
        m.sketches['__profile__'].EllipseByCenterPerimeter(axisPoint1=(7.5, center_y), axisPoint2=(0.0, center_y + 1.5), center=(0.0, center_y))
        m.sketches['__profile__'].Line(point1=(7.5, center_y), point2=(7.5, 0.0))
        m.sketches['__profile__'].VerticalConstraint(addUndoState=
            False, entity=m.sketches['__profile__'].geometry[5])
        m.sketches['__profile__'].Line(point1=(7.5, 0.0), point2=(10.0, 0.0))
        m.sketches['__profile__'].HorizontalConstraint(
            addUndoState=False, entity=
            m.sketches['__profile__'].geometry[6])
        m.sketches['__profile__'].PerpendicularConstraint(
            addUndoState=False, entity1=m.sketches['__profile__'].geometry[5], entity2=
            m.sketches['__profile__'].geometry[6])
        m.sketches['__profile__'].Line(point1=(10.0, 0.0), point2=(10.0, 3.0))
        m.sketches['__profile__'].VerticalConstraint(addUndoState=
            False, entity=m.sketches['__profile__'].geometry[7])
        m.sketches['__profile__'].PerpendicularConstraint(
            addUndoState=False, entity1=
            m.sketches['__profile__'].geometry[6], entity2=
            m.sketches['__profile__'].geometry[7])
        m.sketches['__profile__'].Line(point1=(10.0, 3.0), point2=(0.0, 3.0))
        m.sketches['__profile__'].HorizontalConstraint(
            addUndoState=False, entity=
            m.sketches['__profile__'].geometry[8])
        m.sketches['__profile__'].PerpendicularConstraint(
            addUndoState=False, entity1=
            m.sketches['__profile__'].geometry[7], entity2=
            m.sketches['__profile__'].geometry[8])
        m.sketches['__profile__'].Line(point1=(0.0, 3.0), point2=(0.0, center_y + 1.5))
        m.sketches['__profile__'].VerticalConstraint(addUndoState=
            False, entity=m.sketches['__profile__'].geometry[9])
        m.sketches['__profile__'].PerpendicularConstraint(
            addUndoState=False, entity1=
            m.sketches['__profile__'].geometry[8], entity2=
            m.sketches['__profile__'].geometry[9])
        m.sketches['__profile__'].autoTrimCurve(curve1=
            m.sketches['__profile__'].geometry[3], point1=(
            -7.5, center_y))
        m.sketches['__profile__'].autoTrimCurve(curve1=
            m.sketches['__profile__'].geometry[11], point1=(
            0.5, center_y - 1.5+0.01))
        m.Part(dimensionality=AXISYMMETRIC, name='Part-1', type=
            DEFORMABLE_BODY)
        p = m.parts['Part-1']
        p.BaseShell(sketch=m.sketches['__profile__'])

        p.Set(edges=p.edges.getByBoundingBox(-1,-1,-1, 8,center_y + 1.6,11), name="Pressure")
        p.Set(edges=p.edges.getByBoundingBox(7,-1,-1, 11,11,11), name="Fixed_Some")
        p.Set(edges=p.edges.getByBoundingBox(-1,-1,-1, 7,11,11), name="X0")
        p.SetByBoolean(name='Fixed', sets=(p.sets['Fixed_Some'], p.sets['Pressure']), operation=DIFFERENCE)

        p.setMeshControls(elemShape=QUAD, regions=p.faces.getSequenceFromMask(('[#3 ]', ), ))
        p.seedPart(deviationFactor=0.1, minSizeFactor=0.1, size=0.05)
        p.generateMesh()
        p.Surface(name='Pressure', side1Edges=p.edges.getByBoundingBox(-1,-1,-1, 8,center_y + 1.6,11))
        p.Set(faces=p.faces.getByBoundingBox(-1,-1,-1, 11, 11, 11), name='Part')
        p.setElementType(elemTypes=(ElemType(elemCode=CAX4R, elemLibrary=STANDARD, secondOrderAccuracy=OFF, 
            hourglassControl=DEFAULT, distortionControl=DEFAULT)), regions=p.sets['Part'])
        m.Material(name='Material-1')
        m.materials['Material-1'].Elastic(table=((201669.0, 0.3), 
            ))
        m.materials['Material-1'].Plastic(dataType=PARAMETERS, 
            hardening=COMBINED, scaleStress=None, table=((177.226, 0.0, 0.0), ))
        m.materials['Material-1'].plastic.CyclicHardening(
            parameters=ON, table=((177.226, 986.965, 1.47293), ))
        m.HomogeneousSolidSection(material='Material-1', name=
            'Section-1', thickness=None)
        p.SectionAssignment(offset=0.0, 
            offsetField='', offsetType=MIDDLE_SURFACE, region=
            p.sets['Part'], sectionName='Section-1'
            , thicknessAssignment=FROM_SECTION)
        m.rootAssembly.DatumCsysByThreePoints(coordSysType=
            CYLINDRICAL, origin=(0.0, 0.0, 0.0), point1=(1.0, 0.0, 0.0), point2=(0.0, 
            0.0, -1.0))
        m.rootAssembly.Instance(dependent=ON, name='Part-1-1', 
            part=p)
        m.DisplacementBC(amplitude=UNSET, createStepName='Initial', 
            distributionType=UNIFORM, fieldName='', localCsys=None, name='BC-1', 
            region=m.rootAssembly.instances['Part-1-1'].sets['X0'], 
            u1=SET, u2=UNSET, ur3=UNSET)
        m.DisplacementBC(amplitude=UNSET, createStepName='Initial', 
            distributionType=UNIFORM, fieldName='', localCsys=None, name='BC-2', 
            region=
            m.rootAssembly.instances['Part-1-1'].sets['Fixed'], u1=
            SET, u2=SET, ur3=UNSET)
        m.StaticStep(initialInc=0.1, maxNumInc=10000, name='Step-1'
            , nlgeom=ON, previous='Initial')
        m.Pressure(amplitude=UNSET, createStepName='Step-1', 
            distributionType=UNIFORM, field='', magnitude=19.4, name='Load-1', region=
            m.rootAssembly.instances['Part-1-1'].surfaces['Pressure'])
        job_name = 'Manual%sGeom' % model
        mdb.Job(atTime=None, contactPrint=OFF, description='', echoPrint=OFF, 
            explicitPrecision=SINGLE, getMemoryFromAnalysis=True, historyPrint=OFF, 
            memory=90, memoryUnits=PERCENTAGE, model=model, modelPrint=OFF, 
            multiprocessingMode=DEFAULT, name=job_name, 
            nodalOutputPrecision=SINGLE, numCpus=1, numGPUs=0, numThreadsPerMpiProcess=
            1, queue=None, resultsFormat=ODB, scratch='', type=ANALYSIS, 
            userSubroutine='', waitHours=0, waitMinutes=0)
        mdb.jobs[job_name].writeInput(consistencyChecking=OFF)
    mdb.saveAs(pathName='%s\\%s_M05.cae' % (wk_dir, root))

    new_dir = "%s\\%s_M05" % (main_dir, root)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    for file in os.listdir(main_dir):
            if file.endswith('.inp'):
                shutil.move(os.path.join(main_dir, file), os.path.join(new_dir, file))

generate_models(root, thickness, main_dir=wk_dir)
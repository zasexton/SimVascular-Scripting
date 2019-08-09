# Importing various repos.
import os
from sv import *
import sys
import numpy
from numpy import genfromtxt
import pdb
import re
import math
import os.path
import operator
import copy
import xml.etree.ElementTree as ET
import vtk
import splipy
from splipy.curve_factory import cubic_curve
from splipy.SplineObject import SplineObject

#function to find the radii from an existing contour
def findRadius(name, index):
    contour_group_name = name
    contour_group_name_in_repo = contour_group_name
    contour_ids = [index]

    repo_contour_ids = [contour_group_name_in_repo + "_contour_" + str(id) for id in contour_ids]

    try:
        # Does this item already exist in the Repository?
        if int(Repository.Exists(repo_contour_ids[0])):
            a = 5
        else:
            GUI.ExportContourToRepos(contour_group_name, repo_contour_ids)

        # Calculate the centers of each contour in the segmentation group with a VTK
        # center of mass filter, then calculate the radius of the contour.
        contour_radii = []
        for id in repo_contour_ids:
            # Export the id'th contour to a VTK polyData object.
            contour = Repository.ExportToVtk(id)
            # Apply a VTK filter to locate the center of mass (average) of the points in the contour.
            com_filter = vtk.vtkCenterOfMass()
            com_filter.SetInputData(contour)
            com_filter.Update()
            center = com_filter.GetCenter()

            # Save the points in the contour to a vtkPoints object.
            contour_pts = contour.GetPoints()
            # Iterate through the list of points, but not the last two. (last two are
            #  control points that bung up the solution)
            radii = []
            for point_index in range(contour_pts.GetNumberOfPoints() - 2):
                # Save the point to a cordinate list.
                coord = [0.0, 0.0, 0.0]
                contour_pts.GetPoint(point_index, coord)

                # Compute the "radius" between the current point and the center of the contour.
                # Distance formula: sqrt(dx^2 + dy^2 + dz^2)
                radii.append(math.sqrt(math.pow(coord[0] - center[0], 2) +
                                       math.pow(coord[1] - center[1], 2) +
                                       math.pow(coord[2] - center[2], 2)))

            # Append the average of the "radii" to the list of contour radii as the nominal radius of the current contour.
            contour_radii.append(numpy.mean(radii))

        return (contour_radii[0])



    except Exception as e:
        print("Error!" + str(e))
        return


# The two next functions are 2D geometric function to determine
# if the graft is possible and to shrink the graft vessel if yes.
# Function to determine if a graft can enter the main vessel (eventually shrinked with limits)

def canGraft(vessel,graft,maxShrink):
    shrink=1
    if graft*maxShrink<0.9*vessel:
        while graft*shrink>0.9*vessel:
            shrink-=0.01
        return (True,shrink)
    else:
        return (False,0)



def shrinkGraft(graft_radii, shrink):
    len_graft = len(graft_radii)
    # j'essaie de faire un shrink pertinent
    nb_shrink = int(len_graft / (10 * shrink)) + 1
    factor = (1 - shrink) / nb_shrink
    for k in range(nb_shrink):
        graft_radii[k] *= (shrink + k * factor)



def coarctPipeline(pathList, radiusList, pathName, contourName, modelName, save):
    # Create path --
    p = Path.pyPath() # Shortcut for function call Path.pyPath(), needed when calling SimVascular functions

    # Initializing path
    p.NewObject(pathName)
   # print('Path name: ' + pathName)

    # Adding each point from pathList to created path
    for pathPoint in pathList:
        p.AddPoint(pathPoint)

    # Adding path to repository
    p.CreatePath()

    # Importing created path from repository to the 'Paths' tab in GUI
    GUI.ImportPathFromRepos(pathName,'Paths')

    # Create contour --
    # Initializing variables and creating segmentations (Defaulted to circle)
    Contour.SetContourKernel('Circle')
    pointsLength = len(pathList)
    numEnd = p.GetPathPtsNum() # index at end of pathList
    numSec = int((numEnd-1)/(pointsLength-1))
    polyDataList = []

    # Creating newContourNameList --> going as pathName-ct1, pathName-ct2 etc
    strs = 1 # Random name for keeping index
    newContourNameList = []
    while strs < (pointsLength+1):
        addString = pathName + 'ct' + str(strs)
        newContourNameList.append(addString)
        strs += 1

    # Creating polyDataList --> going as '1ctp', '2ctp' etc
    strs = 1
    polyNameList = []
    while strs < (pointsLength+1):
        addString = str(strs) + 'ctp'
        polyNameList.append(addString)
        strs += 1

    # Creating new contours based on pathList and collecting polyData for modeling
    index = 0
    while index < pointsLength:
        cCall = 'c' + str(index)
        cCall = Contour.pyContour()
        cCall.NewObject(newContourNameList[index], pathName, numSec*index)
        cCall.SetCtrlPtsByRadius(pathList[index], radiusList[index])
        cCall.Create()
        cCall.GetPolyData(polyNameList[index])
        polyDataList.append(polyNameList[index])
        index += 1

    # Importing contours from repository to 'Segmentations' tab in GUI
    GUI.ImportContoursFromRepos(contourName, newContourNameList, pathName, 'Segmentations')

    # Generate model --
    # Initializing variables
    numSegs = 120 # Number of segments defaulted to 120
    srcList = [] # contains SampleLoop generations

    # Apply SampleLoop and appending to srcList for each item of polyDataList
    for item in polyDataList:
        Geom.SampleLoop(item, numSegs, item+'s')
        srcList.append(item+'s')

    pointsLen = len(pathList) # Length of listPathPoints

    # Tangent calls
    t3s = [0, 0, 0]
    tTot = [None]*pointsLen
    calls = 0
    while calls < (pointsLen-1):
        t3s[0] = math.tan(pathList[calls][0])
        t3s[1] = math.tan(pathList[calls][1])
        t3s[2] = math.tan(pathList[calls][2])
        tTot[calls] = t3s
        calls += 1

    # Cosine calls
    c3s = [0, 0, 0]
    cTot = [None]*pointsLen
    calls = 0
    while calls < (pointsLen-1):
        c3s[0] = math.cos(pathList[calls][0])
        c3s[1] = math.cos(pathList[calls][1])
        c3s[2] = math.cos(pathList[calls][2])
        cTot[calls] = c3s
        calls += 1

    # Geom.orientProfile('', x y z, tan(x y z), xyz in plane of obj, 'newOrient')
    # Note: Tan and cos are in degrees, not radians
    dev = 0 # Used to keep index
    while dev < (pointsLen-1):
        st1 = str(dev+1)+'ctps'
        st2 = 'newOrient'+str(dev+1)
        Geom.OrientProfile(str(st1), pathList[dev], tTot[dev], cTot[dev], str(st2))
        dev += 1

    # Creating values to loft solid
    objName = modelName + '_noCap'
    numSegsAlongLength = 240
    numLinearSegsAlongLength = 240
    numModes = 20
    useFFT = 0
    useLinearSampleAlongLength = 1
    Geom.LoftSolid(srcList, objName, numSegs, numSegsAlongLength, numLinearSegsAlongLength, numModes, useFFT, useLinearSampleAlongLength)

    # Importing PolyData from solid to repository (No cap model)
    GUI.ImportPolyDataFromRepos(modelName + '_noCap')

    # Adding caps to model (Full cap model)
    VMTKUtils.Cap_with_ids(modelName + '_noCap', modelName, 0, 0)

    # Shortcut for function call Solid.pySolidModel(), needed when calling SimVascular functions
    s1 = Solid.pySolidModel()

    # Creating model
    Solid.SetKernel('PolyData')
    s1.NewObject('newModel')
    s1.SetVtkPolyData(modelName)
    s1.GetBoundaryFaces(90)
    print("FaceID's found: " + str(s1.GetFaceIds()))
    s1.WriteNative(os.getcwd() + "/" + save + ".vtp")
    GUI.ImportPolyDataFromRepos(modelName)
    return



# function to read a path .pth
def read_centerline(path_name):
    # read in the .pth into a string buffer
    with open(path_name) as f:
        xml = f.read()

    # adjust the structure of the XML data to have a single root node (single top layer tag), as required for ElementTree
    root = ET.fromstring(re.sub(r"(<\?xml[^>]+\?>)", r"\1<root>", xml) + "</root>")

    # access the structure containing all path points
    path_points = root[1][0][0][1]

    # iterate through path point dictionaries and place coordinates into list
    point_list = []
    for point in path_points:
        point_coords = point[0].attrib
        xyz = [float(pos) for pos in [point_coords['x'], point_coords['y'], point_coords['z']]]
        point_list.append(xyz)

    return point_list

# the two next function use the 'curve' class from splipy module of python
# this module isn't in SimVascular and has to be downloaded like Scipy
#(pip install doesn't work)

def curvature(self, t, above=True):
    """curvature(t, above=True)

    Evaluate the curvaure at specified point(s). The curvature is defined as

    .. math:: \\frac{|\\boldsymbol{v}\\times \\boldsymbol{a}|}{|\\boldsymbol{v}|^3}

    :param t: Parametric coordinates in which to evaluate
    :type t: float or [float]
    :param bool above: Evaluation in the limit from above
    :return: Derivative array
    :rtype: numpy.array
    """
    # compute derivative
    v = self.derivative(t, d=1, above=above)
    a = self.derivative(t, d=2, above=above)
    w = numpy.cross(v, a)

    if len(v.shape) == 1:  # single evaluation point
        magnitude = numpy.linalg.norm(w)
        speed = numpy.linalg.norm(v)
    else:  # multiple evaluation points
        if self.dimension == 2:
            magnitude = w  # for 2D-cases np.cross() outputs scalars
        else:  # for 3D, it is vectors
            magnitude = numpy.apply_along_axis(numpy.linalg.norm, -1, w)
        speed = numpy.apply_along_axis(numpy.linalg.norm, -1, v)

    return magnitude / numpy.power(speed, 3)

def torsion(self, t, above=True):
        """torsion(t, above=True)

        Evaluate the torsion for a 3D curve at specified point(s). The torsion is defined as

        .. math:: \\frac{(\\boldsymbol{v}\\times \\boldsymbol{a})\\cdot (d\\boldsymbol{a}/dt)}{|\\boldsymbol{v}\\times \\boldsymbol{a}|^2}

        :param t: Parametric coordinates in which to evaluate
        :type t: float or [float]
        :param bool above: Evaluation in the limit from above
        :return: Derivative array
        :rtype: numpy.array
        """
        if self.dimension == 2:
            # no torsion for 2D curves
            t = ensure_listlike(t)
            return numpy.zeros(len(t))
        elif self.dimension == 3:
            # only allow 3D curves
            pass
        else:
            raise ValueError('dimension must be 2 or 3')

        # compute derivative
        v = self.derivative(t, d=1, above=above)
        a = self.derivative(t, d=2, above=above)
        da = self.derivative(t, d=3, above=above)
        w = numpy.cross(v, a)

        if len(v.shape) == 1:  # single evaluation point
            magnitude = numpy.linalg.norm(w)
            nominator = numpy.dot(w, a)
        else:  # multiple evaluation points
            magnitude = numpy.apply_along_axis(numpy.linalg.norm, -1, w)
            nominator = numpy.array([numpy.dot(w1, da1) for (w1, da1) in zip(w, da)])

        return nominator / numpy.power(magnitude, 2)

############################################
#                   Main                   #
############################################


# contour name of the main vessel
name = 'coarct_contour'
#folder with the path .pth of the main vessel
mainPath = read_centerline('C:\Stanford 2019\Python Project\jeudi1\Paths' + '\\coarct_path.pth')
#pathPoints index on the main vessel Path, where the graft must enter
pointA, pointB = 16, 73

#Graft inputs
number_of_graft_points=50
graftPath = [[0, 0, 0] for k in range(number_of_graft_points)]
graft_rad_list = [1.5 for k in range(number_of_graft_points)]
max_shrink = .3
tgt = numpy.array([[1, 1, 2], [1, -2, 3]])



#finding the radius of the main vessel on the location of the future graft
radius_A=findRadius(name, pointA)
radius_B=findRadius(name, pointB)

#possible is a tuple(Boolean, float) to determine if the graft is possible (True or False) and if True possible[1] is
# the float used to scale the graft vessel radius if it is too big
possible_A = canGraft(radius_A, graft_rad_list[0], max_shrink)

if possible_A[0] :
    shrink_A  = possible_A[1]
    #shrink function shrinks the vessel in a smooth linear maneer at his ends
    shrinkGraft(graft_rad_list, shrink_A)
    reverse_graft_rad_list = copy.deepcopy(graft_rad_list)
    reverse_graft_rad_list.reverse()
    possible_B = canGraft(radius_B, reverse_graft_rad_list[0], max_shrink)

    if possible_B[0]:
        shrink_B = possible_B[1]
        shrinkGraft(reverse_graft_rad_list, shrink_B)

        nb = len(graftPath)
        A=mainPath[pointA]
        B=mainPath[pointB]

        #splipy work
        # (https://pythonhosted.org/Splipy/basic_classes.html#splipy.Curve.curvature)
        coordinates = numpy.array([A, B])
        cb = cubic_curve(x=coordinates, boundary=5, t=None, tangents=tgt)
        bounding = SplineObject.bounding_box(cb)
        x_min, x_max = bounding[0][0], bounding[0][1]
        y_min, y_max = bounding[1][0], bounding[1][1]
        z_min, z_max = bounding[2][0], bounding[2][1]

        fin = SplineObject.end(cb)[0]
        tab = numpy.linspace(0, fin, nb)
        L = list(SplineObject.evaluate(cb, tab))

        curvat = curvature(cb, tab)
        max_curvature = numpy.max(curvat)

        tors = torsion(cb, tab)
        max_torsion = max(numpy.max(tors), -(numpy.min(tors)))

        print('x_min , x_max = ' + str(x_min) + ' , ' + str(x_max))
        print('y_min , y_max = ' + str(y_min) + ' , ' + str(y_max))
        print('z_min , z_max = ' + str(z_min) + ' , ' + str(z_max))
        print('maximal curvature = ' + str(max_curvature))
        print('maximal torsion = ' + str(max_torsion))

        true_graft = [list(x) for x in L]


        # building the graft
        true_rad=copy.deepcopy(reverse_graft_rad_list)
        true_rad.reverse()
        coarctPipeline(true_graft, true_rad, 'graft', 'graft_contour', 'graft_model','graft_model')



        print('successful graft')
        print('location of the graft : ( ' + str(true_graft[0][0])+ ',' + str(true_graft[0][1])+ ','+str(true_graft[0][2])+ ' )' + ' and ' ' ( ' + str(true_graft[-1][0])+ ',' + str(true_graft[-1][1])+ ','+str(true_graft[-1][2])+ ' )')
        print('graft diameter shrink : ' + str(int((1 - shrink_A) * 100)) + '%' + ' and ' + str(int((1 - shrink_B) * 100)) + '%')

    else:
        print('Graft impossible on this point, please choose another location or allow a bigger shrink')
else:
    print('Graft impossible on this point, please choose another location or allow a bigger shrink')






import math
import numpy as np

# These functions generate static paths in the form of a series of waypoints.
# A robot with the FollowPathPolicy will then follow these paths.

# ***************************************************
#
#   Lotzi Bölöni's initial lawnmower implementation
#
# ***************************************************
def generate_lawnmower(x_min, x_max, y_min, y_max, winds):
    """Generates a horizontal lawnmower path on the list
    author:: Lotzi Boloni"""
    current = [x_min, y_min]
    y_step = (y_max - y_min) / (2 * winds)
    path = []
    path.append(current)
    for i in range(winds):
        path.append([x_max, current[1]])
        path.append([x_max, current[1]+y_step])
        path.append([x_min, current[1]+y_step])
        current = [x_min, current[1]+2 * y_step]
        path.append(current)
    path.append([x_max, current[1]])
    return np.array(path)

# ***************************************************
#
#   Sam Matloob's lawnmower implementation with control
#  points version 1
#
# ***************************************************
def generate_lawnmower_with_control_points_v1(x_min, x_max, x_step, y_min, y_max, y_step, control_points, coverage_distance):
    """Generates a lawmower-like pattern with control points
    author:: Sam Matloob (February 2022 ???)"""
    control_points= sorted(control_points, key=lambda x: (x[1], x[0]), reverse= True)
    noEpidemicPoints= len(control_points)
    epidemicPointsToBeAdded=[]
    path = []
    direction=1
    largestX=0.0
    for y in np.arange(y_min, y_max+y_step, y_step):
        if (direction==1):
            for x in np.arange(x_min, x_max + x_step, x_step):
                path.append([x, y])
                while (y <= control_points[noEpidemicPoints-1][1] and x <= control_points[noEpidemicPoints-1][0] and \
                        y + y_step >= control_points[noEpidemicPoints-1][1] and x + x_step >= control_points[noEpidemicPoints-1][0]):
                    if (noEpidemicPoints!=0 and control_points_need_to_be_inserted((x,y), control_points[noEpidemicPoints-1], coverage_distance, x_step, y_step)):
                        epidemicPointsToBeAdded.append(control_points[noEpidemicPoints-1])
                        path.append(control_points[noEpidemicPoints - 1])
                    noEpidemicPoints=noEpidemicPoints-1


                largestX=x
            direction=-1
        else:
            for x in np.arange(largestX, x_min-x_step, -x_step):
                path.append([x, y])
                while (y <= control_points[noEpidemicPoints - 1][1] and x <= control_points[noEpidemicPoints - 1][0] and \
                        y + y_step >= control_points[noEpidemicPoints - 1][1] and x + x_step >=
                        control_points[noEpidemicPoints - 1][0]):

                    if (noEpidemicPoints!=0 and control_points_need_to_be_inserted((x,y), control_points[noEpidemicPoints-1], coverage_distance, x_step, y_step)):
                        epidemicPointsToBeAdded.append(control_points[noEpidemicPoints-1])
                        path.append(control_points[noEpidemicPoints - 1])
                    noEpidemicPoints=noEpidemicPoints-1

            direction=1

    return np.array(path)

def control_points_need_to_be_inserted(path_point, control_point, coverage_distance, x_step, y_step):
    """author Sam Matloob (February 2022?)"""
    if (euclidean_distance(path_point, control_point) >= coverage_distance):
        if (euclidean_distance((path_point[0] + x_step, path_point[1]), control_point) >= coverage_distance):
            if (euclidean_distance((path_point[0], path_point[1] + y_step),
                                           control_point) >= coverage_distance):
                if (euclidean_distance((path_point[0] + x_step, path_point[1] + y_step),
                                               control_point) >= coverage_distance):
                    return True
    return False


def euclidean_distance(point1, point2):
    """Return the euclidian distance between two points"""
    return math.sqrt(math.pow(point2[0]-point1[0],2) + math.pow(point2[1]-point1[1],2))

# ***************************************************
#
#   Sam Matloob's lawnmower implementation with control
#  points version 2
#
# ***************************************************

def generate_lawnmower_path_v2(x_min, x_max, x_step, y_min, y_max, y_step):
    """
    Sam Matloob's lawnmower implementation. The returned path will be modified for the control points
    author Sam Matloob (February 2022?)"""
    path = []
    direction=1
    for y in np.arange(y_min, y_max+y_step, y_step):
        if (direction==1):
            for x in np.arange(x_min, x_max + x_step, x_step):
                path.append([x, y])
            direction=-1
        else:
            for x in np.arange(path[-1][0], x_min-x_step, -x_step):
                path.append([x, y])
            direction = 1
    return path

def add_control_points_v2(lawnmower_path, control_points):
    control_point_mapping={}
    control_points = sorted(control_points, key=lambda x: (x[1], x[0]), reverse=False)
    direction=1
    for control_point in control_points:
        minDistance= [lawnmower_path[0], float('inf')]
        prevLawnmowerPoint=[-1,-1]
        for lawnmowerPoint in lawnmower_path:
            if (prevLawnmowerPoint!=[-1,-1] and lawnmowerPoint[1]!=prevLawnmowerPoint[1]):
                direction= direction*-1
            dist= euclidean_distance(lawnmowerPoint, control_point)
            if (minDistance[1]> dist):
                if (lawnmowerPoint[1]!=prevLawnmowerPoint[1] or direction*(control_point[0] - lawnmowerPoint[0])>=0):
                    minDistance = [lawnmowerPoint, dist]
                else:
                    minDistance = [prevLawnmowerPoint, dist]
            prevLawnmowerPoint= lawnmowerPoint
        if (str(minDistance[0]) in control_point_mapping):
            control_point_mapping[str(minDistance[0])].append(control_point)
        else:
            control_point_mapping[str(minDistance[0])] = [control_point]
    finalPath=[]
    for lawnmowerPoint in lawnmower_path:
        finalPath.append(lawnmowerPoint)
        if str(lawnmowerPoint) in control_point_mapping:
            for epidemicPointMapping in control_point_mapping[str(lawnmowerPoint)]:
                finalPath.append(epidemicPointMapping)
    return finalPath

# ***************************************************
#
#   Partha Datta spiral implementation
#
# ***************************************************

def generate_spiral_path(x_max, y_max, x_min, y_min):
    """Partha Datta's code April 25"""
    y_max = abs(y_max)
    y_min = abs(y_min)
    x_max = abs(x_max)
    x_min = abs(x_min)
    current = [x_min, y_min]
    path = []
    #if (xmax > ymax) :
    i=0
    j=1
    for i in range(x_max//2+1):
        path.append([x_min+i, y_min+i])
        path.append([x_min+i, y_max-i])
        path.append([x_max-i,y_max-i])
        path.append([x_max-i,y_min+i])
        path.append([x_min+j, y_min+i])
        j+=1
    return path

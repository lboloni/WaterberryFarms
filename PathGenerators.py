import math
import numpy as np


def euclidean_distance(point1, point2):
    """Return the euclidian distance between two points"""
    return math.sqrt(math.pow(point2[0]-point1[0],2) + math.pow(point2[1]-point1[1],2))

def get_path_length(starting_point, path):
    """Returns the length of the generated path, assuming that it starts at the starting point"""
    current = starting_point
    length = 0
    for a in path:
        length += euclidean_distance(current, a)
        current = a
    return length



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


def find_fixed_budget_lawnmower(starting_point, x_min, x_max, y_min, y_max, velocity, time):    
    """Finds a lawnmower pattern that best covers the area given a certain budget of time and velocity by performing a binary search on the number of winds. Returns the path"""
    windsmax = 1000
    windsmin = 1
    distancebudget = velocity * time
    while windsmax > windsmin + 1:
        windstest = (windsmin + windsmax) // 2
        # print(windstest)
        path = generate_lawnmower(x_min, x_max, y_min, y_max, winds = windstest)
        length = get_path_length(starting_point, path)
        if length > distancebudget:
            windsmax = windstest
        else:
            windsmin = windstest
    return path


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

    #if direction=1 then the path is going from left to right, if direction=-1 then the path is going from right to left
    direction=1

    # looping through y-axis
    for y in np.arange(y_min, y_max+y_step, y_step):
        if y>y_max: 
            continue
        if (direction==1):

            # looping through x-axis
            for x in np.arange(x_min, x_max + x_step, x_step):
                if x>x_max:
                    continue
                path.append([x, y])
            direction=-1
        else:
            for x in np.arange(path[-1][0], x_min-x_step, -x_step):
                if x<x_min:
                    continue
                path.append([x, y])
            direction = 1
    return path

def add_control_points_v2(lawnmower_path, control_points):
    """
        Sam Matloob's lawnmower implementation. The returned path will be modified for the control points
        author Sam Matloob (February 2022?)"""
    # this method will add the control points to the lawnmower_path

    # control_point_mapping is a dictionary variable that stores the lawnmower_path point as a key, while the value
    # is a list of the closest control_points to the lawnmower_path point.
    control_point_mapping={}
    control_points = sorted(control_points, key=lambda x: (x[1], x[0]), reverse=False)

    # looping through the control_points
    for control_point in control_points:
        minDistance= [lawnmower_path[0], float('inf')]
        prevLawnmowerPoint=[-1,-1]
        direction=1

        # looping through the lawnmower_path and check which point in that path is the closest to the control point
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

    # the following loop will insert all control points to the lawnmower_point using the dictionary variable "control_point_mapping"
    # that was built above
    for lawnmowerPoint in lawnmower_path:
        finalPath.append(lawnmowerPoint)
        if str(lawnmowerPoint) in control_point_mapping:
            # I initially thought I will need to sort the list of points here closest to farthest from the lawnmower point
            # But, I think sorting is not needed since I'm already sorting the control points above.
            #sorted(control_point_mapping[str(lawnmowerPoint)], key= lambda x: euclidean_distance((float(str(lawnmowerPoint).split('[')[1].split(',')[0]), float(str(lawnmowerPoint).split(',')[1].split(']')[0])), x))

            for epidemicPointMapping in control_point_mapping[str(lawnmowerPoint)]:
                finalPath.append(list(epidemicPointMapping))
    return finalPath

def find_fixed_budget_lawnmower_v2(control_points, starting_point, x_min, x_max, y_min, y_max, velocity, time):    
    """Finds a lawnmower pattern that best covers the area given a certain budget of time and velocity by performing a binary search on the number of winds. Returns the path"""
    step_max = float(y_max-y_min)
    step_min = 0.1
    distancebudget = velocity * time
    while step_max > step_min + 0.05: # 0.05 is the minimum vertical step
        step_test = (step_min + step_max) / 2
        pathWithNoControlPoints= generate_lawnmower_path_v2(x_min, x_max, 1, y_min, y_max, step_test)
        pathWithControlPoints= add_control_points_v2(pathWithNoControlPoints, control_points)
        path = np.array(pathWithControlPoints)
        length = get_path_length(starting_point, path)
        if length > distancebudget:
            step_min = step_test  
        else:
            step_max = step_test
                
    return path

# ***************************************************
#
#   Partha Datta spiral implementation
#
# ***************************************************

def generate_spiral_path_ParthaDatta(x_max, y_max, x_min, y_min):
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

# ***************************************************
#
#   Lotzi's implementation for a square spiral path
#
# ***************************************************
def generate_spiral_path(x_min, x_max, y_min, y_max, step = 1):
    """A square spiral coverage of a rectangular area with a specific step size"""
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    cur_x_min = x_min 
    cur_x_max = x_max
    cur_y_min = y_min
    cur_y_max = y_max
    cur_x = x_min
    cur_y = y_min
    phase = 0
    path = []
    done = False
    while not done:
        if phase == 0:
            cur_x = cur_x_min
            cur_y = cur_y_min
            cur_x_min = cur_x_min + step
            if cur_x_min > center_x:
                done = True
        if phase == 1:
            cur_x = cur_x_max
            cur_y = cur_y_min
            cur_y_min = cur_y_min + step
            if cur_y_min > center_y:
                done = True
        if phase == 2:
            cur_x = cur_x_max
            cur_y = cur_y_max
            cur_x_max = cur_x_max - step
            if cur_x_max < center_x:
                done = True
        if phase == 3:
            cur_x = cur_x_min
            cur_y = cur_y_max
            # only here??
            cur_y_max = cur_y_max - step
            if cur_y_max < center_y:
                done = True
        phase = (phase + 1) % 4
        path.append([cur_x, cur_y])
    return np.array(path)

def find_fixed_budget_spiral(starting_point, x_min, x_max, y_min, y_max, velocity, time):    
    """Finds a spiral pattern that best covers the area given a certain budget of time and velocity by performing a binary search on the number of winds. Returns the path"""
    step_max = 1000.0
    step_min = 1.0
    distancebudget = velocity * time
    while step_max > step_min + 1:
        step_test = (step_min + step_max) / 2
        path = generate_spiral_path(x_min, x_max, y_min, y_max, step = step_test)
        length = get_path_length(starting_point, path)
        if length > distancebudget:
            step_min = step_test
        else:
            step_max = step_test        
    return path


import networkx as nx
from networkx.algorithms import approximation
import math

def compute(data):
    G= nx.complete_graph(len(data))
    for u in range(len(data)):
        for v in range(len(data)):
            if (u!=v):
                G[u][v]['weight']=euclidean_distance(data[u], data[v])
    
    path= approximation.christofides(G)
    modifiedPath=[]
    """ Construct a new path """
    for vertex in path[:-1]:
        modifiedPath.append(data[vertex])
    return modifiedPath
    
def build_graph(data):
    graph = [[0 for x in range(len(data))] for y in range(len(data))] 

    for this in range(len(data)):
        for another_point in range(len(data)):
            #if this != another_point and this <= another_point:
            #    graph[this][another_point] = get_length(data[this][0], data[this][1], data[another_point][0],
            #                                            data[another_point][1])
            graph[this][another_point] = get_length(data[this][0], data[this][1], data[another_point][0],
                                                        data[another_point][1])
    
    return graph

def get_length(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (1.0 / 2.0)

def euclidean_distance(point1, point2):
    """Return the euclidian distance between two points"""
    return math.sqrt(math.pow(point2[0]-point1[0],2) + math.pow(point2[1]-point1[1],2))

#print(compute([(1,1), (2,2), (10,10), (5,5), (15,5), (0,5), (7,0)]))

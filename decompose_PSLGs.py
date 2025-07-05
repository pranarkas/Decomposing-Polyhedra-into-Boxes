from shapely.geometry import LineString, box, Point, MultiPoint
from shapely.ops import split
import matplotlib.pyplot as plt
import itertools
import random
import os
import json
from intervaltree import Interval, IntervalTree
import bisect
import networkx as nx
from itertools import combinations

#Part 1: Read a file in Instances which contains a collection of rectangles and then make it a Planar Straight-Line Graph (PSLG)

## Step 1: Read rectangles

### Dimensions of the large bounding box -- this is for drawing outer box for decomposition. NOTE: CHANGE ON BOTH PROGRAMS IF NEEDED
box_width = 100
box_height = 100

def read_rectangles_from_file(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return [box(*coords[0], *coords[2]) for coords in data]  # rebuild using two corners

folder = "Instances"
all_files = [f for f in os.listdir(folder)]
chosen_file = random.choice(all_files) #Choose this if you want a random instance
#chosen_file = "instance_079.json"  #nice looking instance: 079, 136, 147, 192, 748, 764, 946
print(chosen_file)
filename = os.path.join(folder, chosen_file)

rectangles = read_rectangles_from_file(filename)

###add the bounding box at the end
rectangles.append(box(*[0,0],*[box_height,box_width]))

## Step 2: Extract edges from rectangles
G = nx.Graph() #This will hold our PSLG. Each node will have a position attribute (pos) while each edge will have a is_vertical boolean attribute 

for r in rectangles:
    coords = list(r.exterior.coords)
    for i in range(len(coords) - 1):  # we have coords[i+1]
        v1 = (int(coords[i][0]),int(coords[i][1]))
        v2 = (int(coords[i+1][0]),int(coords[i+1][1]))
        G.add_node(v1, pos = coords[i]) 
        G.add_node(v2, pos = coords[i+1])
        G.add_edge(v1, v2, is_vertical = (v1[0] == v2[0])) 

### Note that after this step there are no vertices where the rectangles intersect each other. However, if two rectangles happen to share a vertex, this will "collate" them into one

## Step 3: Find all intersection points and compute the PSLG
def get_vertical_and_horizontal_edges(G): #compute and return the vertical and horizontal edges of G; vertical edges are sorted from left to right, horizontal edges are sorted from bottom to top
    vertical_edges = []
    horizontal_edges = []
    for e in G.edges(data = True): #extract edges of G along with the data
        u, v, data = e
        if data.get("is_vertical") == True:
            if u[1] > v[1]: #sort the tuple of a vertical edge by y-coordinate
                u, v = v, u
            vertical_edges.append((u,v))
        else:
            if u[0] > v[0]: #sort the tuple of a horizontal edge by x-coordinate
                u, v = v, u
            horizontal_edges.append((u,v))
    horizontal_edges.sort(key=lambda edge: edge[0][1]) #sort the horizontal edges by y-coordinate
    vertical_edges.sort(key=lambda edge: edge[0][0]) #sort the vertical edges by x-coordinate
    return [vertical_edges,horizontal_edges]

def resolve_high_degree_vertices(G):
    for v in G.nodes():
        if G.degree(v) > 2: 
            top = []
            bottom = []
            left = []
            right = []
            x0, y0 = v  # Coordinates of v

            for nbr in G.neighbors(v): #classify neighbors into top, bottom, left, or right
                x1, y1 = nbr

                if y1 > y0:
                    top.append(nbr)
                elif y1 < y0:
                    bottom.append(nbr)
                elif x1 < x0:
                    left.append(nbr)
                elif x1 > x0:
                    right.append(nbr)

            # Sort each list by x or y coordinate depending on direction
            top.sort(key=lambda x: x[1])     # Sort by y
            bottom.sort(key=lambda x: x[1])  # Sort by y
            left.sort(key=lambda x: x[0])    # Sort by x
            right.sort(key=lambda x: x[0])   # Sort by x

            list_of_neighbors = [top,bottom,left,right]

            for coll in list_of_neighbors: #for each of these for sets, check if the there is more than one neighbor there; if make this collection of edges into a chain instead 
                if len(coll) > 1:
                    for nbr in coll: #remove all the neighbors
                        G.remove_edge(v,nbr)
                    
                    G.add_edge(v,coll[0]) #add the first edge
                    for i in range(len(coll) - 1):
                        G.add_edge(coll[i],coll[i+1],is_vertical = (coll[i][0] == coll[i+1][0])) #add edge to the previous node


def find_all_intersections(G): #this will compute the intersections between the rectangles and make these vertices
    vertical_edges, horizontal_edges = get_vertical_and_horizontal_edges(G)

    for e1 in vertical_edges: #run through the vertical edges; recall that they are sorted from left to right
        for i, e2 in enumerate(horizontal_edges): #run through the horizontal edges; recall that they are sorted from top to bottom
            edges_adjacent = bool(set(e1) & set(e2)) 
            if not edges_adjacent: #if the edges are adjacent we have nothing to do
                seg1 = LineString([e1[0], e1[1]])
                seg2 = LineString([e2[0], e2[1]]) #convert them into Shapely segments for intersection-type queries
                if seg1.crosses(seg2): #if they cross each other (i.e. their interiors intersect) then we delete these two edges and have 4 in their place after we make the intersection point a new node
                    inter = seg1.intersection(seg2)
                    new_node = (int(inter.x), int(inter.y))

                    G.add_node(new_node, pos = (inter.x, inter.y))
                    G.add_edge(new_node,e1[0],is_vertical = True)
                    G.add_edge(new_node,e1[1],is_vertical = True)
                    G.add_edge(new_node,e2[0],is_vertical = False)
                    G.add_edge(new_node,e2[1],is_vertical = False)
                    
                    if G.has_edge(e1[0],e1[1]):
                        G.remove_edge(e1[0],e1[1])
                    else:
                        print("Error in crossing: (e1, e2) = ", e1, e2)
                    
                    if G.has_edge(e2[0],e2[1]):
                        G.remove_edge(e2[0],e2[1])    
                    else:
                        print("Error in crossing: (e1, e2) = ", e1, e2)
                        
                    e1 = (new_node,e1[1]) #now, note that the (now deleted) edge e1 could be intersected again by some other horizontal segment. This segment is above e2 because of our sorting. So we replace e1 with the upper portion after this intersection.
                    horizontal_edges[i] = (new_node,e2[1]) #similarly, the edge e2 could intersect more vertical edges. These edges are to the right. So, we replace e2 with the right part after the intersection.
                elif seg1.touches(seg2): #if they only touch each other (i.e., form a "T"), then we have no new nodes; just three new edges in place of one
                    inter = seg1.intersection(seg2)
                    node = (int(inter.x), int(inter.y))
                    if seg1.touches(Point(node)): #if one of the end points of e1 is in the interior of e2, then we delete e2 and add two new edges
                        G.add_edge(node,e2[0],is_vertical = False)
                        G.add_edge(node,e2[1],is_vertical = False)
                        if G.has_edge(e2[0],e2[1]):
                            G.remove_edge(e2[0],e2[1])    
                        else:
                            print("Error in touching: (e1, e2) = ", e1, e2)
                        horizontal_edges[i] = (node,e2[1])  #we do the same trick for horizontal edges as before; note that we do not need to worry about the vertical edge here since it "ends" at e2
                    if seg2.touches(Point(node)):
                        G.add_edge(node,e1[0],is_vertical = True)
                        G.add_edge(node,e1[1],is_vertical = True)
                        if G.has_edge(e1[0],e1[1]):
                            G.remove_edge(e1[0],e1[1])    
                        else:
                            print("Error in touching: (e1, e2) = ", e1, e2)
                        e1 = (node, e1[1]) #same trick for vertical edges as before

resolve_high_degree_vertices(G) #deal with high degree vertices in G

find_all_intersections(G) #compute the intersections and make the PSLG

fig, ax = plt.subplots() #Plot the input PSLG
pos = nx.get_node_attributes(G, 'pos') # Extract positions from node attributes
nx.draw_networkx(G, pos, with_labels=False, node_color='blue', node_size=10, font_size = 8)
plt.gca().set_aspect('equal')
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(False)

#Part 2: Compute the decomposition

vertical_edges, horizontal_edges = get_vertical_and_horizontal_edges(G) #recompute the vertical and horizontal edges

def create_interval_tree(edges, is_vertical_tree): #takes (Polyline, 1/0) -> 1 is for vertical edges, 0 for horizontal
    edges_interval = [(a[is_vertical_tree],b[is_vertical_tree]) for (a,b) in edges] #for vertical edges, it will create the interval based on y-coordinates; for horizontal edges, it will do so based on x-coordinates
    edges_tuple_with_index = [(x, y+0.1, (i,edges[i][0][1-is_vertical_tree])) for i, (x, y) in enumerate(edges_interval)] #make it a list of tuples with index and x/y coordinate to call back; note that the 0.1 is because the data structure considers the interval to be of the form [x,y)
    return IntervalTree.from_tuples(edges_tuple_with_index) #Build interval tree

vertical_edges_tree = create_interval_tree(vertical_edges, is_vertical_tree=1) #create the trees for vertical and horizontal edges. Note that for the "simple" decomposition, we only need the vertical tree. We need both for the optimal decomposition.
horizontal_edges_tree = create_interval_tree(horizontal_edges, is_vertical_tree=0)

def horizontal_ray_shooting(point, vertical_edges_tree, is_left): #given a point, the vertical edges in the interval tree and the direction of ray shooting, this tells us the point of intersection and the edge it intersects; note that we do not modify the graph here, we do that later in add_ray_points_to_graph 
    x_point, y_point = point
    intersecting_edge_index = sorted([seg.data[0] for seg in vertical_edges_tree[y_point]]) #vertical_edges_tree[y_point] tells us the segments that the horizontal ray through point intersects; sort the indices and store it here
    intersecting_edge_coordinates = sorted([seg.data[1] for seg in vertical_edges_tree[y_point]]) #sort the x-coordinates of the vertical segments and store those here; note that the sorted order is the same in the two lists since we have the segments sorted by x-coordinate
    cutoff = bisect.bisect_left(intersecting_edge_coordinates, x_point) #compute the index in which the point will be inserted to in this list
    if is_left:
        intersection_point = (intersecting_edge_coordinates[cutoff - 1],y_point) #if you are shooting a ray to the left, then it intersects the segment which is in position cutoff-1
        bisected_edge_index = intersecting_edge_index[cutoff - 1] 
    else:
        intersection_point = (intersecting_edge_coordinates[cutoff + 1],y_point)#if you are shooting a ray to the left, then it intersects the segment which is in position cutoff+1
        bisected_edge_index = intersecting_edge_index[cutoff + 1]

    return (intersection_point,bisected_edge_index) #return the intersection point and the index of the bisected edge; note that we have not modified the graph here

def is_left_reflex(v): #this checks if a vertex v is left-reflex -- this is true only when it has degree 2 and one neighbor is to the right 
    flag = False
    if G.degree(v) == 2:
        u, w = G.neighbors(v)
        if v[0] < u[0] or v[0] < w[0]:
            flag = True
    return flag 

def is_right_reflex(v): #this checks if a vertex is right-reflex -- this is true only when it has degree 2 and one neighbor is to the left 
    flag = False
    if G.degree(v) == 2:
        u, w = G.neighbors(v)
        if v[0] > u[0] or v[0] > w[0]:
            flag = True
    return flag

def add_ray_points_to_graph(vertical_edges, ray_shooting_queries): #given the rays and the vertical edges, we now modify the graph; note that the rays are sorted by the edges they intersect and within that they are sorted by y-coordinate
    i = 0
    while i < len(ray_shooting_queries): #iterate through the rays
        edge_index = ray_shooting_queries[i][2] #this stores the index of the edge that is being bisected
        bisected_edge = vertical_edges[edge_index] #this is the edge that is bisected
        while ray_shooting_queries[i][2] == edge_index: #run through the internal loop as long as we are bisecting the same edge
            G.add_node(ray_shooting_queries[i][1],pos = ray_shooting_queries[i][1]) #add the intersection point as a node and then add an edge between the intersection point and the source of the ray
            G.add_edge(ray_shooting_queries[i][0],ray_shooting_queries[i][1], is_vertical = 0)

            if not (ray_shooting_queries[i][1] == bisected_edge[0] or ray_shooting_queries[i][1] == bisected_edge[1]): #if the ray ends up at an endpoint of the bisected edge, there is nothing to do
                G.add_edge(ray_shooting_queries[i][1], bisected_edge[0], is_vertical = 1) #otherwise, we need to divide that edge into two and delete the original edge
                G.add_edge(ray_shooting_queries[i][1], bisected_edge[1], is_vertical = 1)

                if G.has_edge(bisected_edge[0],bisected_edge[1]):
                    G.remove_edge(bisected_edge[0],bisected_edge[1]) 
                else:
                    print("Error: (vertex, intersection point, edge) = ", ray_shooting_queries[i][0], ray_shooting_queries[i][1], vertical_edges[ray_shooting_queries[i][2]], bisected_edge, "\n\n")
            else:
                print("Degeneracy: (vertex, intersection point, edge) = ", ray_shooting_queries[i][0], ray_shooting_queries[i][1], vertical_edges[ray_shooting_queries[i][2]], "\n\n")

            bisected_edge = (ray_shooting_queries[i][1],bisected_edge[1]) #Update the bisected edge to be top part of the edge; this works since the other rays that hit the edge are above because of sorting
            i = i +1
            if i >= len(ray_shooting_queries):
                return

left_reflex = []
right_reflex = []

for v in G.nodes(): ###Compute the left reflex vertices
    if is_left_reflex(v):
        left_reflex.append(v)

ray_shooting_queries = []

for v in left_reflex: ### for the left_reflex vertices, shoot rays to the left
    if v[0] > 0:
        intersection_point, bisected_edge_index = horizontal_ray_shooting(v, vertical_edges_tree, is_left = 1)
        data = (v, intersection_point, bisected_edge_index)
        ray_shooting_queries.append(data)

ray_shooting_queries.sort(key=lambda data: (data[2], data[1][1])) ###sort by the edge index and then by the y-coordinate of the intersection point

add_ray_points_to_graph(vertical_edges, ray_shooting_queries) # add the rays to the graph

vertical_edges, horizontal_edges = get_vertical_and_horizontal_edges(G) #recompute the vertical and horizontal edges and then the trees (note that the indices of the edges in the tree are now out of order -- indeed some of the edges in the tree are no longer present in the graph)
vertical_edges_tree = create_interval_tree(vertical_edges, is_vertical_tree=1)
horizontal_edges_tree = create_interval_tree(horizontal_edges, is_vertical_tree=0)

for v in G.nodes(): ###Compute the right reflex vertices
    if is_right_reflex(v):
        right_reflex.append(v)

ray_shooting_queries = []

for v in right_reflex: ### for the left_reflex vertices, shoot rays to the right
    if v[0] < box_width:
        intersection_point, bisected_edge_index = horizontal_ray_shooting(v, vertical_edges_tree, is_left = 0)
        data = (v, intersection_point, bisected_edge_index)
        ray_shooting_queries.append(data)

ray_shooting_queries.sort(key=lambda data: (data[2], data[1][1])) ###sort by the edge index and then by the y-coordinate of the intersection point

add_ray_points_to_graph(vertical_edges, ray_shooting_queries)

#Part 3: Plot the PSLG and the decomposition
pos = nx.get_node_attributes(G, 'pos') #Extract positions from node attributes

fig, ax = plt.subplots()
nx.draw_networkx(G, pos, with_labels=False, node_color='blue', node_size=10, font_size = 8)
plt.gca().set_aspect('equal')

for v in left_reflex:
    ax.plot(v[0], v[1], 'ro', markersize=4)

for v in right_reflex:
    ax.plot(v[0], v[1], 'go', markersize=4)

ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(False)

plt.show()

e1 = ((0,0),(0,100))
e2 = ((0,50),(50,50))

seg1 = LineString([e1[0], e1[1]])
seg2 = LineString([e2[0], e2[1]])
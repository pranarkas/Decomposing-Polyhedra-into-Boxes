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
from networkx.algorithms import bipartite

#Part 1: Read a file in Instances which contains a collection of rectangles and then make it a Planar Straight-Line Graph (PSLG)

### Dimensions of the large bounding box -- this is for drawing outer box for decomposition. NOTE: CHANGE ON BOTH PROGRAMS IF NEEDED
box_width = 100
box_height = 100

## Step 1: Read rectangles

def read_rectangles_from_file(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return [box(*coords[0], *coords[2]) for coords in data]  # rebuild using two corners

folder = "Instances"
all_files = [f for f in os.listdir(folder)]
chosen_file = random.choice(all_files) #Choose this if you want a random instance
#chosen_file = "instance_140.json"  #nice looking instance: 079, 136, 147, 152, 192, 384*, 436*, 702*, 748, 764, 836, 946

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
                    
                    G.add_edge(v,coll[0],is_vertical = (v[0] == coll[0][0])) #add the first edge
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

def is_bottom_reflex(v): #this checks if a vertex is bottom-reflex -- this is true only when it has degree 2 and one neighbor is to the top 
    flag = False
    if G.degree(v) == 2:
        u, w = G.neighbors(v)
        if v[1] < u[1] or v[1] < w[1]:
            flag = True
    return flag

def is_top_reflex(v): #this checks if a vertex is top-reflex -- this is true only when it has degree 2 and one neighbor is to the bottom 
    flag = False
    if G.degree(v) == 2:
        u, w = G.neighbors(v)
        if v[1] > u[1] or v[1] > w[1]:
            flag = True
    return flag

resolve_high_degree_vertices(G) #deal with high degree vertices in G
find_all_intersections(G) #compute the intersections and make the PSLG

vertices_with_reflexivity_data = [((v[0],v[1]),is_left_reflex(v),is_right_reflex(v),is_bottom_reflex(v),is_top_reflex(v)) for v in G.nodes()] #collect the vertices along with reflexivity data for plotting in different colors
left_bottom_reflex = [v[0] for v in vertices_with_reflexivity_data if (v[1] and v[3])]
right_bottom_reflex = [v[0] for v in vertices_with_reflexivity_data if (v[2] and v[3])]
left_top_reflex = [v[0] for v in vertices_with_reflexivity_data if (v[1] and v[4])]
right_top_reflex = [v[0] for v in vertices_with_reflexivity_data if (v[2] and v[4])]

fig, ax = plt.subplots() #Plot the input PSLG
pos = nx.get_node_attributes(G, 'pos') # Extract positions from node attributes
nx.draw_networkx(G, pos, with_labels=False, node_color='blue', node_size=10, font_size = 8)
if left_bottom_reflex:
    xs, ys = zip(*left_bottom_reflex)
    plt.scatter(xs, ys, color='red', zorder = 3, s = 8)

if right_bottom_reflex:
    xs, ys = zip(*right_bottom_reflex)
    plt.scatter(xs, ys, color='orange', zorder = 3, s = 8)

if left_top_reflex:
    xs, ys = zip(*left_top_reflex)
    plt.scatter(xs, ys, color='green', zorder = 3, s = 8)

if right_top_reflex:
    xs, ys = zip(*right_top_reflex)
    plt.scatter(xs, ys, color='brown', zorder = 3, s = 8)
plt.gca().set_aspect('equal')
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(False)

#Part 2: Compute the decomposition

##Step 1: Compute and add a maximum independent set of good edges to G
def create_interval_tree(edges, is_vertical_tree): #takes (Polyline, 1/0) -> 1 is for vertical edges, 0 for horizontal
    edges_interval = [(a[is_vertical_tree],b[is_vertical_tree]) for (a,b) in edges] #for vertical edges, it will create the interval based on y-coordinates; for horizontal edges, it will do so based on x-coordinates
    edges_tuple_with_index = [(x, y+0.1, (i,edges[i][0][1-is_vertical_tree])) for i, (x, y) in enumerate(edges_interval)] #make it a list of tuples with index and x/y coordinate to call back; note that the 0.1 is because the data structure considers the interval to be of the form [x,y)
    return IntervalTree.from_tuples(edges_tuple_with_index) #Build interval tree

def ray_shooting(point, tree, is_left, is_down, is_ray_horizontal): #given a point, the vertical/horizontal edges in the interval tree and the direction of ray shooting, this tells us the point of intersection and the edge it intersects; note that we do not modify the graph here, we do that later in add_ray_points_to_graph 
    x_point, y_point = point
    
    if is_ray_horizontal: #if the ray shooting is parallel to the x-axis,
        coordinate_for_intersection_check = y_point #the ray will have constant y-coordinate
        coordinate_for_cutoff_check = x_point #to compute the segment which has been intersected, we must know the points x-coordinate
        direction = is_left #the direction we care about is if it is to the left or it is down
    else: #if the ray shooting is parallel to the y-axis,
        coordinate_for_intersection_check = x_point #the ray will have constant x-coordinate
        coordinate_for_cutoff_check = y_point #to compute the segment which has been intersected, we must know the points x-coordinate
        direction = is_down #the direction we care about is if it is to the left or it is down
    
    intersecting_edge_index = sorted([seg.data[0] for seg in tree[coordinate_for_intersection_check]]) #tree[coordinate_for_intersection_check] tells us the segments that the ray through point intersects; sort the indices of these segments and store it here
    intersecting_edge_coordinates = sorted([seg.data[1] for seg in tree[coordinate_for_intersection_check]]) #sort the x/y-coordinates of the vertical/horizontal segments and store those here; note that the sorted order is the same in the two lists since we have the segments sorted by x/y-coordinates
    cutoff = bisect.bisect_left(intersecting_edge_coordinates, coordinate_for_cutoff_check) #compute the index in which the point will be inserted to in this list
    
    if direction: #if you are shooting a ray to the left or below
        intersection_point = [0,0]
        intersection_point[1 - is_ray_horizontal] = intersecting_edge_coordinates[cutoff - 1] #if you are shooting vertically/horizontally, then the intersection point has the same x/y coordinate
        intersection_point[is_ray_horizontal] = coordinate_for_intersection_check #if you are shooting a ray to the left or below, then it intersects the segment which is in position cutoff-1
        intersection_point = tuple(intersection_point)
        bisected_edge_index = intersecting_edge_index[cutoff - 1] 
    else:#if you are shooting a ray to the right or above
        intersection_point = [0,0]
        intersection_point[1 - is_ray_horizontal] = intersecting_edge_coordinates[cutoff + 1] #if you are shooting vertically/horizontally, then the intersection point has the same x/y coordinate
        intersection_point[is_ray_horizontal] = coordinate_for_intersection_check #if you are shooting a ray to the right or above, then it intersects the segment which is in position cutoff + 1
        intersection_point = tuple(intersection_point)
        bisected_edge_index = intersecting_edge_index[cutoff + 1]

    return (intersection_point,bisected_edge_index) #return the intersection point and the index of the bisected edge; note that we have not modified the graph here

def compute_good_edges(vertices_with_reflexivity_data, vertical_edges, horizontal_edges, vertical_edges_tree, horizontal_edges_tree): #this computes the good edges -- i.e., horizontal or vertical chords of the PSLG between its reflex vertices
    good_edges = []
    for vertex_with_reflexivity_data in vertices_with_reflexivity_data:
        v, is_left, is_right, is_bottom, is_top = vertex_with_reflexivity_data #take the vertex as well as the data about what type of reflex vertex it is
        if is_left: #if it is a left-reflex vertex, shoot a ray to the left; if the ray hits another vertex, then this is a good edge
            if v[0] > 0:
                intersection_point, bisected_edge_index = ray_shooting(point = v, tree = vertical_edges_tree, is_left = 1, is_down = 0, is_ray_horizontal = 1)
                bisected_edge = vertical_edges[bisected_edge_index]
                if (intersection_point == bisected_edge[0] or intersection_point == bisected_edge[1]):
                    good_edges.append((v,intersection_point))
        elif is_right: #if it is a right-reflex vertex, shoot a ray to the right; if the ray hits another vertex, then this is a good edge
            if v[0] < box_width:
                intersection_point, bisected_edge_index = ray_shooting(point = v, tree = vertical_edges_tree, is_left = 0, is_down = 0, is_ray_horizontal = 1)
                bisected_edge = vertical_edges[bisected_edge_index]
                if (intersection_point == bisected_edge[0] or intersection_point == bisected_edge[1]):
                    good_edges.append((v,intersection_point))
        if is_bottom: #if it is a bottom-reflex vertex, shoot a ray below; if the ray hits another vertex, then this is a good edge
            if v[1] > 0:
                intersection_point, bisected_edge_index = ray_shooting(point = v, tree = horizontal_edges_tree, is_left = 0, is_down = 1, is_ray_horizontal = 0)
                bisected_edge = horizontal_edges[bisected_edge_index]
                if (intersection_point == bisected_edge[0] or intersection_point == bisected_edge[1]):
                    good_edges.append((v,intersection_point))
        elif is_top: #if it is a top-reflex vertex, shoot a ray to the above; if the ray hits another vertex, then this is a good edge
            if v[1] < box_height:
                intersection_point, bisected_edge_index = ray_shooting(point = v, tree = horizontal_edges_tree, is_left = 0, is_down = 0, is_ray_horizontal = 0)
                bisected_edge = horizontal_edges[bisected_edge_index]
                if (intersection_point == bisected_edge[0] or intersection_point == bisected_edge[1]):
                    good_edges.append((v,intersection_point))
    
    deduped_edges = set(tuple(sorted(edge)) for edge in good_edges) #note that good_edges, at the moment, has edges (a,b) and (b,a); de-duplicate them and store it back in good_edges
    good_edges = list(deduped_edges)
    good_edges = [(edge, edge[0][0] == edge[1][0]) for edge in good_edges] #also include information about if the edge is vertical or not

    return good_edges

def find_max_IS_of_good_edges(good_edges): #given a collection of good edges, we compute the maximum subset where no to edges intersect each other
    H = nx.Graph()
    for edge in good_edges:
        H.add_node(edge[0],bipartite = edge[1]) #H has a node for each good edge; the bipartition we define is horizontal edges on one side and vertical edges on the other 

    top_nodes = {n for n, d in H.nodes(data=True) if d["bipartite"] == 0} #these are the horizontal edges
    bottom_nodes = {n for n, d in H.nodes(data=True) if d["bipartite"] == 1} #these are the vertical edges

    for e1 in top_nodes:
        for e2 in bottom_nodes:
            seg1 = LineString([e1[0], e1[1]])
            seg2 = LineString([e2[0], e2[1]]) #convert them into Shapely segments for intersection-type queries
            if seg1.intersects(seg2): #add an edge between the nodes if their corresponding segments intersect
                H.add_edge(e1,e2)

    matching = bipartite.maximum_matching(H, top_nodes = top_nodes) #compute the maximum matching
    min_vertex_cover = bipartite.to_vertex_cover(H, matching=matching, top_nodes = top_nodes) #use this to compute a min Vertex Cover
    max_independent_set = set(H.nodes()) - min_vertex_cover #The complement of the vertex cover is the independent set
    return max_independent_set    

want_optimal = True #select True if you want the optimal decomposition; false if you want a 2--approx
independent_good_edges = []

if want_optimal:
    vertical_edges, horizontal_edges = get_vertical_and_horizontal_edges(G) #recompute the vertical and horizontal edges
    vertical_edges_tree = create_interval_tree(vertical_edges, is_vertical_tree=1) #create the trees for vertical and horizontal edges. Note that for the "simple" decomposition, we only need the vertical tree. We need both for the optimal decomposition.
    horizontal_edges_tree = create_interval_tree(horizontal_edges, is_vertical_tree=0)

    vertices_with_reflexivity_data = [((v[0],v[1]),is_left_reflex(v),is_right_reflex(v),is_bottom_reflex(v),is_top_reflex(v)) for v in G.nodes()]

    good_edges = compute_good_edges(vertices_with_reflexivity_data, vertical_edges, horizontal_edges, vertical_edges_tree, horizontal_edges_tree)

    if(len(good_edges) > 0):
        independent_good_edges = find_max_IS_of_good_edges(good_edges)
        for edge in independent_good_edges:
            G.add_edge(edge[0],edge[1], is_vertical = (edge[0][0] == edge[1][0]))
        vertical_edges, horizontal_edges = get_vertical_and_horizontal_edges(G) #recompute the vertical and horizontal edges
        vertical_edges_tree = create_interval_tree(vertical_edges, is_vertical_tree=1) #create the trees for vertical and horizontal edges. Note that for the "simple" decomposition, we only need the vertical tree. We need both for the optimal decomposition.
        horizontal_edges_tree = create_interval_tree(horizontal_edges, is_vertical_tree=0)

    print("Maximum Independent Set of good edges:", independent_good_edges)

## Step 2: Perform vertical or horizontal decomposition

def add_ray_points_to_graph(edges, ray_shooting_queries, is_ray_horizontal): #given the rays and the vertical edges, we now modify the graph; note that the rays are sorted by the edges they intersect and within that they are sorted by y-coordinate
    i = 0
    while i < len(ray_shooting_queries): #iterate through the rays
        edge_index = ray_shooting_queries[i][2] #this stores the index of the edge that is being bisected
        bisected_edge = edges[edge_index] #this is the edge that is bisected
        while ray_shooting_queries[i][2] == edge_index: #run through the internal loop as long as we are bisecting the same edge
            G.add_node(ray_shooting_queries[i][1],pos = ray_shooting_queries[i][1]) #add the intersection point as a node and then add an edge between the intersection point and the source of the ray
            G.add_edge(ray_shooting_queries[i][0],ray_shooting_queries[i][1], is_vertical = is_ray_horizontal)

            if not (ray_shooting_queries[i][1] == bisected_edge[0] or ray_shooting_queries[i][1] == bisected_edge[1]): #if the ray ends up at an endpoint of the bisected edge, there is nothing to do
                G.add_edge(ray_shooting_queries[i][1], bisected_edge[0], is_vertical = is_ray_horizontal) #otherwise, we need to divide that edge into two and delete the original edge
                G.add_edge(ray_shooting_queries[i][1], bisected_edge[1], is_vertical = is_ray_horizontal)

                if G.has_edge(bisected_edge[0],bisected_edge[1]):
                    G.remove_edge(bisected_edge[0],bisected_edge[1]) 
                else:
                    print("Error: (vertex, intersection point, edge) = ", ray_shooting_queries[i][0], ray_shooting_queries[i][1], edges[ray_shooting_queries[i][2]], bisected_edge, "\n\n")

            bisected_edge = (ray_shooting_queries[i][1],bisected_edge[1]) #Update the bisected edge to be top/right part of the edge; this works since the other rays that hit the edge are above because of sorting
            i = i +1
            if i >= len(ray_shooting_queries):
                return

def ray_shooting_from_a_set_of_reflex_vertices(reflex_set, tree, is_ray_horizontal, is_left, is_down, box_dimensions, is_vertical_decomposition, is_first):
    
    ray_shooting_queries = []
    if is_first: #when we are dealing with the second set of vertices, the inequality direction needs to be flipped
        inequality_direction = 1
    else:
        inequality_direction = -1
    
    for v in reflex_set: ### for the bottom_reflex/left_reflex vertices, shoot rays to the bottom
        if v[is_vertical_decomposition]*inequality_direction > box_dimensions*inequality_direction:
            intersection_point, bisected_edge_index = ray_shooting(point = v, tree = tree, is_left = is_left, is_down = is_down, is_ray_horizontal = is_ray_horizontal)
            data = (v, intersection_point, bisected_edge_index)
            ray_shooting_queries.append(data)
    
    ray_shooting_queries.sort(key=lambda data: (data[2], data[1][1 - is_vertical_decomposition])) ###sort by the edge index and then by the y-coordinate of the intersection point

    return ray_shooting_queries

def vertical_or_horizontal_decomposition(G, is_vertical_decomposition):
    vertical_edges, horizontal_edges = get_vertical_and_horizontal_edges(G) #recompute the vertical and horizontal edges and then the trees (note that the indices of the edges in the tree are now out of order -- indeed some of the edges in the tree are no longer present in the graph)
    vertical_edges_tree = create_interval_tree(vertical_edges, is_vertical_tree=1)
    horizontal_edges_tree = create_interval_tree(horizontal_edges, is_vertical_tree=0)

    vertices_with_reflexivity_data = [((v[0],v[1]),is_left_reflex(v),is_right_reflex(v),is_bottom_reflex(v),is_top_reflex(v)) for v in G.nodes()]

    if is_vertical_decomposition: #if this is a vertical decomposition, then look at bottom_reflex vertices and shoot down
        first_reflex_set = [data[0] for data in vertices_with_reflexivity_data if data[3] == 1]
        tree = horizontal_edges_tree
        edges = horizontal_edges
        is_ray_horizontal = 0
        is_left = 0
        is_down = 1
        box_dimensions = 0

    else:
        first_reflex_set = [data[0] for data in vertices_with_reflexivity_data if data[1] == 1] #if this is a horizontal decomposition, then look at left_reflex vertices and shoot left
        tree = vertical_edges_tree
        edges = vertical_edges
        is_ray_horizontal = 1
        is_left = 1
        is_down = 0
        box_dimensions = 0
    
    ray_shooting_queries = ray_shooting_from_a_set_of_reflex_vertices(first_reflex_set, tree, is_ray_horizontal, is_left, is_down, box_dimensions, is_vertical_decomposition, True)#compute the rays
    add_ray_points_to_graph(edges, ray_shooting_queries, is_ray_horizontal=is_ray_horizontal) # add the rays to the graph

    vertical_edges, horizontal_edges = get_vertical_and_horizontal_edges(G) #recompute the vertical and horizontal edges and then the trees (note that the indices of the edges in the tree are now out of order -- indeed some of the edges in the tree are no longer present in the graph)
    vertical_edges_tree = create_interval_tree(vertical_edges, is_vertical_tree=1)
    horizontal_edges_tree = create_interval_tree(horizontal_edges, is_vertical_tree=0)

    vertices_with_reflexivity_data = [((v[0],v[1]),is_left_reflex(v),is_right_reflex(v),is_bottom_reflex(v),is_top_reflex(v)) for v in G.nodes()]

    if is_vertical_decomposition: #if this is a vertical decomposition, then look at top_reflex vertices and shoot up
        second_reflex_set = [data[0] for data in vertices_with_reflexivity_data if data[4] == 1]
        tree = horizontal_edges_tree
        edges = horizontal_edges
        is_ray_horizontal = 0
        is_left = 0
        is_down = 0
        box_dimensions = box_height

    else: #if this is a horizontal decomposition, then look at right_reflex vertices and shoot right
        second_reflex_set = [data[0] for data in vertices_with_reflexivity_data if data[2] == 1]
        tree = vertical_edges_tree
        edges = vertical_edges
        is_ray_horizontal = 1
        is_left = 0
        is_down = 0
        box_dimensions = box_width
    
    ray_shooting_queries = ray_shooting_from_a_set_of_reflex_vertices(second_reflex_set, tree, is_ray_horizontal, is_left, is_down, box_dimensions, is_vertical_decomposition, False)#compute the rays
    add_ray_points_to_graph(edges, ray_shooting_queries, is_ray_horizontal=is_ray_horizontal) # add the rays to the graph


is_vertical_decomposition = True #choose if you want to do a vertical or horizontal decomposition
vertical_or_horizontal_decomposition(G,is_vertical_decomposition)

#Part 3: Plot the PSLG and the decomposition
pos = nx.get_node_attributes(G, 'pos') #Extract positions from node attributes

fig, ax = plt.subplots()
nx.draw_networkx(G, pos, with_labels=False, node_color='blue', node_size=10, font_size = 8)

if want_optimal:
    nx.draw_networkx_edges(G, pos, edgelist=independent_good_edges, edge_color='purple', width=3)

if left_bottom_reflex:
    xs, ys = zip(*left_bottom_reflex)
    plt.scatter(xs, ys, color='red', zorder = 3, s = 8)

if right_bottom_reflex:
    xs, ys = zip(*right_bottom_reflex)
    plt.scatter(xs, ys, color='orange', zorder = 3, s = 8)

if left_top_reflex:
    xs, ys = zip(*left_top_reflex)
    plt.scatter(xs, ys, color='green', zorder = 3, s = 8)

if right_top_reflex:
    xs, ys = zip(*right_top_reflex)
    plt.scatter(xs, ys, color='brown', zorder = 3, s = 8)

plt.gca().set_aspect('equal')

ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(False)

plt.show()
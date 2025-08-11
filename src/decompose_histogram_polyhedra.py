from shapely.geometry import LineString
import matplotlib.pyplot as plt
import random
import os
import networkx as nx
from itertools import combinations
from networkx.algorithms import bipartite
from orthogonal_ray_shooting import create_interval_tree, ray_shooting, get_vertical_and_horizontal_edges
import orthogonal_dcel

#Part 1: Read a file in Histogram_Polyhedra_Instances; this is an orthogonal DCEL; we convert it to a networkX graph

### Dimensions of the large bounding box -- this is for drawing outer box for decomposition. NOTE: CHANGE ON BOTH PROGRAMS IF NEEDED
box_width = 100
box_height = 100

## Step 1: Read instance

is_pkl = True

if is_pkl:
    folder = "Histogram_Polyhedra_Instances_pkl"

else:
    folder = "Histogram_Polyhedra_Instances_json"

all_files = [f for f in os.listdir(folder)]
chosen_file = random.choice(all_files) #Choose this if you want a random instance
#chosen_file = "instance_196_003.json" #nice pickle instances: 985, 074, 672, 253, 146 (one rectangle), 444, 839, 896, 856 ; nice json instances: 417_005, 981, 895, 919, 320, 539, 028, 588

print(chosen_file)

filename = os.path.join(folder, chosen_file)

if is_pkl:
    dcel = (orthogonal_dcel.DCEL).load_from_pickle(filename=filename)

else:
    dcel = (orthogonal_dcel.DCEL).load_from_json(filename=filename)

half_edge_height_dict = {} #this holds the heights of the edges for the decomposition

for he in dcel.half_edges: #add the half edges of the dcel to the dictionary along with the heights of their corresponding face
    he_coords = (he.origin.coords, he.destination.coords)
    half_edge_height_dict[he_coords] = he.face.height

print("\n\nInput DCEL")

for face in dcel.faces:
    print(face, "height: ", face.height)

G = nx.Graph()
G = dcel.convert_to_graph()

##Step 2: Plot the input PSLG and 3D instance
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

def draw_graph(left_bottom_reflex, right_bottom_reflex, left_top_reflex, right_top_reflex, G, fig, ax):
    pos = nx.get_node_attributes(G, 'pos') # Extract positions from node attributes
    nx.draw_networkx(G, pos, with_labels=False, node_color='gray', node_size=10, font_size = 8)
    if left_bottom_reflex:
        xs, ys = zip(*left_bottom_reflex)
        plt.scatter(xs, ys, color='red', zorder = 3, s = 15)

    if right_bottom_reflex:
        xs, ys = zip(*right_bottom_reflex)
        plt.scatter(xs, ys, color='orange', zorder = 3, s = 15)

    if left_top_reflex:
        xs, ys = zip(*left_top_reflex)
        plt.scatter(xs, ys, color='cyan', zorder = 3, s = 15)

    if right_top_reflex:
        xs, ys = zip(*right_top_reflex)
        plt.scatter(xs, ys, color='lime', zorder = 3, s = 15)

    plt.gca().set_aspect('equal')
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)

vertical_edges_for_dcel, horizontal_edges_for_dcel = get_vertical_and_horizontal_edges(G) #compute the vertical and horizontal edges
vertical_edges_tree_for_dcel = create_interval_tree(vertical_edges_for_dcel, is_vertical_tree=1) #create the trees for vertical and horizontal edges. Note that for the "simple" decomposition, we only need the vertical tree. We need both for the optimal decomposition.
horizontal_edges_tree_for_dcel = create_interval_tree(horizontal_edges_for_dcel, is_vertical_tree=0)

dcel.plot_histogram_polyhedron() #plot the  3D instance

vertices_with_reflexivity_data = [((v[0],v[1]),is_left_reflex(v),is_right_reflex(v),is_bottom_reflex(v),is_top_reflex(v)) for v in G.nodes()] #collect the vertices along with reflexivity data for plotting in different colors
left_bottom_reflex = [v[0] for v in vertices_with_reflexivity_data if (v[1] and v[3])]
right_bottom_reflex = [v[0] for v in vertices_with_reflexivity_data if (v[2] and v[3])]
left_top_reflex = [v[0] for v in vertices_with_reflexivity_data if (v[1] and v[4])]
right_top_reflex = [v[0] for v in vertices_with_reflexivity_data if (v[2] and v[4])]

fig, ax = plt.subplots() #Plot the input PSLG
draw_graph(left_bottom_reflex, right_bottom_reflex, left_top_reflex, right_top_reflex, G, fig, ax)

#Part 2: Compute the decomposition

##Step 1: Compute and add a maximum independent set of good edges to G
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

def add_new_edge_to_half_edge_dictionary(edge, height_1 = None, height_2 = None): #given an undirected edge uv, add u->v with height_1 and v->u with height2 to the dictionary if they are given; otherwise, compute the height by a point in dcel face query
    v1 = (edge[0][0], edge[0][1])
    v2 = (edge[1][0], edge[1][1])

    if height_1 == None: #if the heights are not given, compute the face that has the midpoint of this edge and report its height; in this case the edge will be in the interior of a face and therefore both half edges have the same height
        point = ((edge[0][0] + edge[1][0])/2, (edge[0][1] + edge[1][1])/2)
        face_with_point = dcel.face_with_point(point, edges = vertical_edges_for_dcel, tree = vertical_edges_tree_for_dcel, is_ray_horizontal = True)
        height_1 = face_with_point.height
        height_2 = height_1

    half_edge_height_dict[(v1,v2)] = height_1 #add the heights to the dictionary
    half_edge_height_dict[(v2,v1)] = height_2

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
            add_new_edge_to_half_edge_dictionary(edge) #this edge is in the interior of the face, so we need to compute the height through the midpoint of the edge
            G.add_edge(edge[0],edge[1], is_vertical = (edge[0][0] == edge[1][0]))
        vertical_edges, horizontal_edges = get_vertical_and_horizontal_edges(G) #recompute the vertical and horizontal edges
        vertical_edges_tree = create_interval_tree(vertical_edges, is_vertical_tree=1) #create the trees for vertical and horizontal edges. Note that for the "simple" decomposition, we only need the vertical tree. We need both for the optimal decomposition.
        horizontal_edges_tree = create_interval_tree(horizontal_edges, is_vertical_tree=0)

    print("\n\nMaximum Independent Set of good edges:", independent_good_edges, "\n")
## Step 2: Perform vertical or horizontal decomposition
def remove_edge_from_dictionary(edge):
    v1 = (edge[0][0], edge[0][1])
    v2 = (edge[1][0], edge[1][1])

    del half_edge_height_dict[(v1, v2)] 
    del half_edge_height_dict[(v2, v1)]

def add_ray_points_to_graph(edges, ray_shooting_queries, is_ray_horizontal): #given the rays and the vertical edges, we now modify the graph; note that the rays are sorted by the edges they intersect and within that they are sorted by y-coordinate
    i = 0
    while i < len(ray_shooting_queries): #iterate through the rays
        edge_index = ray_shooting_queries[i][2] #this stores the index of the edge that is being bisected
        bisected_edge = edges[edge_index] #this is the edge that is bisected

        he1 = bisected_edge #directed half edge
        he2 = (bisected_edge[1], bisected_edge[0]) #the opposite directed half edge

        height1 = half_edge_height_dict[he1] #pick out the heights of the bisected edge
        height2 = half_edge_height_dict[he2]

        new_edge = (ray_shooting_queries[i][0],ray_shooting_queries[i][1]) #this is the new edge perpendicular to the bisected edge; note that all edges that intersect this edge all are within the same face of the dcel

        point = ((new_edge[0][0] + new_edge[1][0])/2, (new_edge[0][1] + new_edge[1][1])/2)
        face_with_point = dcel.face_with_point(point, edges = vertical_edges_for_dcel, tree = vertical_edges_tree_for_dcel, is_ray_horizontal = True)
        height = face_with_point.height #compute the height using the midpoint of this edge

        while ray_shooting_queries[i][2] == edge_index: #run through the internal loop as long as we are bisecting the same edge
            G.add_node(ray_shooting_queries[i][1],pos = ray_shooting_queries[i][1]) #add the intersection point as a node and then add an edge between the intersection point and the source of the ray
            G.add_edge(ray_shooting_queries[i][0],ray_shooting_queries[i][1], is_vertical = bool(1 - is_ray_horizontal))

            add_new_edge_to_half_edge_dictionary((ray_shooting_queries[i][0],ray_shooting_queries[i][1]),height, height)

            if not (ray_shooting_queries[i][1] == bisected_edge[0] or ray_shooting_queries[i][1] == bisected_edge[1]): #if the ray ends up at an endpoint of the bisected edge, there is nothing to do
                G.add_edge(bisected_edge[0], ray_shooting_queries[i][1], is_vertical = bool(is_ray_horizontal)) #otherwise, we need to divide that edge into two and delete the original edge
                G.add_edge(ray_shooting_queries[i][1], bisected_edge[1], is_vertical = bool(is_ray_horizontal))

                add_new_edge_to_half_edge_dictionary((bisected_edge[0], ray_shooting_queries[i][1]), height1, height2)
                add_new_edge_to_half_edge_dictionary((ray_shooting_queries[i][1], bisected_edge[1]), height1, height2)

                if G.has_edge(bisected_edge[0],bisected_edge[1]):
                    G.remove_edge(bisected_edge[0],bisected_edge[1])
                    remove_edge_from_dictionary(bisected_edge)
                    
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

#Part 3: Plot the PSLG decomposition
pos = nx.get_node_attributes(G, 'pos') #Extract positions from node attributes

fig, ax = plt.subplots()
draw_graph(left_bottom_reflex, right_bottom_reflex, left_top_reflex, right_top_reflex, G, fig, ax)

#Part 4: Plot the new DCEL
dcel_decomposed = orthogonal_dcel.DCEL()
dcel_decomposed.compute_faces_from_graph(G)

for face in dcel_decomposed.faces:
    if face.is_external:
        face.height = 0
    else:
        he = face.start_half_edge
        he_coords = (he.origin.coords, he.destination.coords)
        height = half_edge_height_dict[he_coords]
        face.height = height

dcel_decomposed.s = dcel.s
dcel_decomposed.t = dcel.t

vertical_edges, _ = get_vertical_and_horizontal_edges(G) #recompute the vertical and horizontal edges and then the trees (note that the indices of the edges in the tree are now out of order -- indeed some of the edges in the tree are no longer present in the graph)
vertical_edges_tree = create_interval_tree(vertical_edges, is_vertical_tree=1)

#Part 5: Get the GCS graph and plot it on top of the new DCEL

H = dcel_decomposed.get_graph_for_GCS(edges = vertical_edges, tree=vertical_edges_tree) #get the GCS graph

# Calculate positions
pos = {}
for vertex in H.nodes():
    if isinstance(vertex, tuple) and len(vertex) == 2:
        # Edge vertex: use midpoint
        p1, p2 = vertex
        pos[vertex] = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    else:
        pos[vertex] = (vertex[0], vertex[1])

# Draw the graph
nx.draw_networkx_edges(H, pos, alpha=0.5, edge_color='gray', ax=ax)

# Draw different vertex types with different colors
edge_vertices = [v for v in H.nodes() if isinstance(v, tuple) and len(v) == 2]
source_vertices = [v for v in H.nodes() if H.nodes[v].get('is_source', False)]
dest_vertices = [v for v in H.nodes() if H.nodes[v].get('is_destination', False)]

if edge_vertices:
    nx.draw_networkx_nodes(H, pos, nodelist=edge_vertices, node_color='black', node_size=10, alpha=0.7, ax=ax)
if source_vertices:
    nx.draw_networkx_nodes(H, pos, nodelist=source_vertices, node_color='blue', node_size=10, node_shape='s', ax=ax)
if dest_vertices:
    nx.draw_networkx_nodes(H, pos, nodelist=dest_vertices, node_color='darkgreen', node_size=10, node_shape='s', ax=ax)

print("Source:",dcel.s, "\nDestination:", dcel.t)
# print("\n\nFinal DCEL")

# for face in dcel_decomposed.faces:
#     print(face, "height: ", face.height)

dcel_decomposed.plot_histogram_polyhedron()

plt.show()

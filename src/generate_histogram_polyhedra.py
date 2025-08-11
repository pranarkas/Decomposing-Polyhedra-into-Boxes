import random
from shapely.geometry import box
import os
import networkx as nx
from orthogonal_ray_shooting import (
    create_interval_tree,
    ray_shooting,
    get_vertical_and_horizontal_edges,
)
from shapely.geometry import LineString, box, Point, MultiPoint
import orthogonal_dcel


"""
import click # argparse, typer
parse command line arguments 
validate command line arguments 
during validation, add additional dependenet fields into the cliparse 
pass that into the main function 
"""

# add CLI for is_pkl, box dimensions

# Pickle or JSON?
is_pkl = False

# Dimensions of the large bounding box
box_width = 100
box_length = 100
box_height = 100

# Number of rectangles to generate
max_number_of_rectangles = 10

# Size constraints for small rectangles
min_width = 1
max_width = box_width / 2
min_length = 1
max_length = box_length / 2
min_height = 1
max_height = box_height


def generate_random_rectangle():
    width = random.randint(min_width, max_width)
    length = random.randint(min_length, max_length)
    x = random.randint(0, box_width - width)
    y = random.randint(0, box_length - length)
    return box(x, y, x + width, y + length)


def resolve_high_degree_vertices(G):
    for v in G.nodes():
        if G.degree(v) > 2:
            top = []
            bottom = []
            left = []
            right = []
            x0, y0 = v  # Coordinates of v

            for nbr in G.neighbors(
                v
            ):  # classify neighbors into top, bottom, left, or right
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
            top.sort(key=lambda x: x[1])  # Sort by y
            bottom.sort(key=lambda x: x[1])  # Sort by y
            left.sort(key=lambda x: x[0])  # Sort by x
            right.sort(key=lambda x: x[0])  # Sort by x

            list_of_neighbors = [top, bottom, left, right]

            for coll in list_of_neighbors:  # for each of these for sets, check if the there is more than one neighbor there; if make this collection of edges into a chain instead
                if len(coll) > 1:
                    for nbr in coll:  # remove all the neighbors
                        G.remove_edge(v, nbr)

                    G.add_edge(
                        v, coll[0], is_vertical=(v[0] == coll[0][0])
                    )  # add the first edge
                    for i in range(len(coll) - 1):
                        G.add_edge(
                            coll[i],
                            coll[i + 1],
                            is_vertical=(coll[i][0] == coll[i + 1][0]),
                        )  # add edge to the previous node


def find_all_intersections(
    G,
):  # this will compute the intersections between the rectangles and make these vertices
    vertical_edges, horizontal_edges = get_vertical_and_horizontal_edges(G)

    for e1 in (
        vertical_edges
    ):  # run through the vertical edges; recall that they are sorted from left to right
        for i, e2 in enumerate(
            horizontal_edges
        ):  # run through the horizontal edges; recall that they are sorted from top to bottom
            edges_adjacent = bool(set(e1) & set(e2))
            if not edges_adjacent:  # if the edges are adjacent we have nothing to do
                seg1 = LineString([e1[0], e1[1]])
                seg2 = LineString(
                    [e2[0], e2[1]]
                )  # convert them into Shapely segments for intersection-type queries
                if seg1.crosses(
                    seg2
                ):  # if they cross each other (i.e. their interiors intersect) then we delete these two edges and have 4 in their place after we make the intersection point a new node
                    inter = seg1.intersection(seg2)
                    new_node = (int(inter.x), int(inter.y))

                    G.add_node(new_node, pos=(inter.x, inter.y))
                    G.add_edge(new_node, e1[0], is_vertical=True)
                    G.add_edge(new_node, e1[1], is_vertical=True)
                    G.add_edge(new_node, e2[0], is_vertical=False)
                    G.add_edge(new_node, e2[1], is_vertical=False)

                    if G.has_edge(e1[0], e1[1]):
                        G.remove_edge(e1[0], e1[1])
                    else:
                        print("Error in crossing: (e1, e2) = ", e1, e2)

                    if G.has_edge(e2[0], e2[1]):
                        G.remove_edge(e2[0], e2[1])
                    else:
                        print("Error in crossing: (e1, e2) = ", e1, e2)

                    e1 = (
                        new_node,
                        e1[1],
                    )  # now, note that the (now deleted) edge e1 could be intersected again by some other horizontal segment. This segment is above e2 because of our sorting. So we replace e1 with the upper portion after this intersection.
                    horizontal_edges[i] = (
                        new_node,
                        e2[1],
                    )  # similarly, the edge e2 could intersect more vertical edges. These edges are to the right. So, we replace e2 with the right part after the intersection.
                elif seg1.touches(
                    seg2
                ):  # if they only touch each other (i.e., form a "T"), then we have no new nodes; just three new edges in place of one
                    inter = seg1.intersection(seg2)
                    node = (int(inter.x), int(inter.y))
                    if seg1.touches(
                        Point(node)
                    ):  # if one of the end points of e1 is in the interior of e2, then we delete e2 and add two new edges
                        G.add_edge(node, e2[0], is_vertical=False)
                        G.add_edge(node, e2[1], is_vertical=False)
                        if G.has_edge(e2[0], e2[1]):
                            G.remove_edge(e2[0], e2[1])
                        else:
                            print("Error in touching: (e1, e2) = ", e1, e2)
                        horizontal_edges[i] = (
                            node,
                            e2[1],
                        )  # we do the same trick for horizontal edges as before; note that we do not need to worry about the vertical edge here since it "ends" at e2
                    if seg2.touches(Point(node)):
                        G.add_edge(node, e1[0], is_vertical=True)
                        G.add_edge(node, e1[1], is_vertical=True)
                        if G.has_edge(e1[0], e1[1]):
                            G.remove_edge(e1[0], e1[1])
                        else:
                            print("Error in touching: (e1, e2) = ", e1, e2)
                        e1 = (node, e1[1])  # same trick for vertical edges as before


def convert_to_PSLG(rectangles) -> nx.Graph:
    G = nx.Graph()  # This will hold our PSLG. Each node will have a position attribute (pos) while each edge will have a is_vertical boolean attribute

    for r in rectangles:
        coords = list(r.exterior.coords)
        for i in range(len(coords) - 1):  # we have coords[i+1]
            v1 = (int(coords[i][0]), int(coords[i][1]))
            v2 = (int(coords[i + 1][0]), int(coords[i + 1][1]))
            G.add_node(v1, pos=coords[i])
            G.add_node(v2, pos=coords[i + 1])
            G.add_edge(v1, v2, is_vertical=(v1[0] == v2[0]))

    resolve_high_degree_vertices(G)  # deal with high degree vertices in G
    find_all_intersections(G)  # compute the intersections and make the PSLG

    return G


def generate_random_point_in_dcel(dcel: orthogonal_dcel.DCEL, edges, tree):
    x = random.randint(0, box_width)
    y = random.randint(0, box_height)
    point = (x, y)
    face = dcel.face_with_point(point, edges=edges, tree=tree, is_ray_horizontal=True)
    height = face.height
    z = random.randint(0, height)
    point = [x, y, z]
    return point


if is_pkl:
    output_dir = "Histogram_Polyhedra_Instances_pkl"
else:
    output_dir = "Histogram_Polyhedra_Instances_json"

num_instances = 999
num_s_t_pairs = 9

for i in range(1, num_instances + 1):
    num_rectangles = random.randint(1, max_number_of_rectangles)
    rectangles = []
    while len(rectangles) < num_rectangles:  # generate random rectangles
        new_rectangle = generate_random_rectangle()
        rectangles.append(new_rectangle)

    rectangles.append(
        box(*[0, 0], *[box_width, box_length])
    )  # add a large bounding box

    G = convert_to_PSLG(rectangles)  # convert this into a PSLG

    dcel = orthogonal_dcel.DCEL()
    dcel.compute_faces_from_graph(G)  # make this a dcel

    print(i, dcel)

    vertical_edges, _ = get_vertical_and_horizontal_edges(G)
    tree = create_interval_tree(edges=vertical_edges, is_vertical_tree=True)

    for face in dcel.faces:
        face.height = random.randint(min_height, max_height)
        if face.is_external:
            face.height = 0

    for j in range(1, num_s_t_pairs + 1):
        s = generate_random_point_in_dcel(dcel=dcel, edges=vertical_edges, tree=tree)
        t = generate_random_point_in_dcel(dcel=dcel, edges=vertical_edges, tree=tree)
        dcel.s = s
        dcel.t = t
        if is_pkl:
            filename = os.path.join(output_dir, f"instance_{i:03}_{j:03}.pkl")
            dcel.save_to_pickle(filename)  # we use pickle to save it as a binary file
        else:
            filename = os.path.join(output_dir, f"instance_{i:03}_{j:03}.json")
            dcel.save_to_json(filename)  # we use json to save it as a readable file


def main():
    pass


if __name__ == "__main__":
    main()

from intervaltree import Interval, IntervalTree
import bisect


def get_vertical_and_horizontal_edges(
    G,
):  # compute and return the vertical and horizontal edges of G; vertical edges are sorted from left to right, horizontal edges are sorted from bottom to top
    vertical_edges = []
    horizontal_edges = []
    for e in G.edges(data=True):  # extract edges of G along with the data
        u, v, data = e
        if data.get("is_vertical") == True:
            if u[1] > v[1]:  # sort the tuple of a vertical edge by y-coordinate
                u, v = v, u
            vertical_edges.append((u, v))
        else:
            if u[0] > v[0]:  # sort the tuple of a horizontal edge by x-coordinate
                u, v = v, u
            horizontal_edges.append((u, v))
    horizontal_edges.sort(
        key=lambda edge: edge[0][1]
    )  # sort the horizontal edges by y-coordinate
    vertical_edges.sort(
        key=lambda edge: edge[0][0]
    )  # sort the vertical edges by x-coordinate
    return [vertical_edges, horizontal_edges]


def create_interval_tree(
    edges, is_vertical_tree
):  # takes (Polyline, 1/0) -> 1 is for vertical edges, 0 for horizontal
    edges_interval = [
        (a[is_vertical_tree], b[is_vertical_tree]) for (a, b) in edges
    ]  # for vertical edges, it will create the interval based on y-coordinates; for horizontal edges, it will do so based on x-coordinates
    edges_tuple_with_index = [
        (x, y + 0.1, (i, edges[i][0][1 - is_vertical_tree]))
        for i, (x, y) in enumerate(edges_interval)
    ]  # make it a list of tuples with index and x/y coordinate to call back; note that the 0.1 is because the data structure considers the interval to be of the form [x,y)
    return IntervalTree.from_tuples(edges_tuple_with_index)  # Build interval tree


def ray_shooting(
    point, tree, is_left, is_down, is_ray_horizontal
):  # given a point, the vertical/horizontal edges in the interval tree and the direction of ray shooting, this tells us the point of intersection and the edge it intersects; note that we do not modify the graph here, we do that later in add_ray_points_to_graph
    x_point, y_point = point

    if is_ray_horizontal:  # if the ray shooting is parallel to the x-axis,
        coordinate_for_intersection_check = (
            y_point  # the ray will have constant y-coordinate
        )
        coordinate_for_cutoff_check = x_point  # to compute the segment which has been intersected, we must know the points x-coordinate
        direction = (
            is_left  # the direction we care about is if it is to the left or it is down
        )
    else:  # if the ray shooting is parallel to the y-axis,
        coordinate_for_intersection_check = (
            x_point  # the ray will have constant x-coordinate
        )
        coordinate_for_cutoff_check = y_point  # to compute the segment which has been intersected, we must know the points x-coordinate
        direction = (
            is_down  # the direction we care about is if it is to the left or it is down
        )

    intersecting_edge_index = sorted(
        [seg.data[0] for seg in tree[coordinate_for_intersection_check]]
    )  # tree[coordinate_for_intersection_check] tells us the segments that the ray through point intersects; sort the indices of these segments and store it here
    intersecting_edge_coordinates = sorted(
        [seg.data[1] for seg in tree[coordinate_for_intersection_check]]
    )  # sort the x/y-coordinates of the vertical/horizontal segments and store those here; note that the sorted order is the same in the two lists since we have the segments sorted by x/y-coordinates
    cutoff = bisect.bisect_left(
        intersecting_edge_coordinates, coordinate_for_cutoff_check
    )  # compute the index in which the point will be inserted to in this list

    if direction:  # if you are shooting a ray to the left or below
        intersection_point = [0, 0]
        intersection_point[1 - is_ray_horizontal] = intersecting_edge_coordinates[
            cutoff - 1
        ]  # if you are shooting vertically/horizontally, then the intersection point has the same x/y coordinate
        intersection_point[is_ray_horizontal] = (
            coordinate_for_intersection_check  # if you are shooting a ray to the left or below, then it intersects the segment which is in position cutoff-1
        )
        intersection_point = tuple(intersection_point)
        bisected_edge_index = intersecting_edge_index[cutoff - 1]
    else:  # if you are shooting a ray to the right or above
        intersection_point = [0, 0]
        intersection_point[1 - is_ray_horizontal] = intersecting_edge_coordinates[
            cutoff + 1
        ]  # if you are shooting vertically/horizontally, then the intersection point has the same x/y coordinate
        intersection_point[is_ray_horizontal] = (
            coordinate_for_intersection_check  # if you are shooting a ray to the right or above, then it intersects the segment which is in position cutoff + 1
        )
        intersection_point = tuple(intersection_point)
        bisected_edge_index = intersecting_edge_index[cutoff + 1]

    return (
        intersection_point,
        bisected_edge_index,
    )  # return the intersection point and the index of the bisected edge; note that we have not modified the graph here

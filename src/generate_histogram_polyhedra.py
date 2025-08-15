"""
Generate random histogram polyhedra instances and serialize them.

This script builds 2D rectilinear arrangements from random axis-aligned
rectangles, converts the arrangement into a PSLG (planar straight-line
graph), then into an orthogonal DCEL with face heights. For each instance,
it also samples random source/target 3D points that lie within faces (z
bounded by the face height) and writes each instance to disk (pickle or
JSON).

Key steps:
- Random rectangles inside a fixed bounding box.
- Build PSLG with vertical/horizontal edges, split at intersections.
- Convert to orthogonal DCEL and assign random heights to faces.
- Sample s/t points within faces using ray-shooting location.

Note: There is a placeholder for adding a CLI. For now, parameters are
configured by constants below. If you need a CLI, consider `argparse` or
`click` and thread parsed values into the main loop.
"""

import random
from shapely.geometry import box
import os
import networkx as nx
import argparse
from format_logger import setup_logger
import logging
from orthogonal_ray_shooting import (
    create_interval_tree,
    ray_shooting,
    get_vertical_and_horizontal_edges,
)
from shapely.geometry import LineString, box, Point, MultiPoint, Polygon
import orthogonal_dcel

logger = logging.getLogger(__name__)


def generate_random_rectangle(args) -> Polygon:
    """Create a random axis-aligned rectangle inside the bounding box.

    Returns a Shapely `Polygon` (via `box`) whose coordinates are integers
    within the configured width/length limits.
    """
    width = random.randint(args.min_width, args.max_width)
    length = random.randint(args.min_length, args.max_length)
    # Bottom-left corner chosen so the rectangle stays inside the bounds.
    x = random.randint(0, args.box_width - width)
    y = random.randint(0, args.box_length - length)
    return box(x, y, x + width, y + length)


def resolve_high_degree_vertices(G: nx.Graph) -> None:
    """Transform degree>2 vertices into local chains per cardinal direction.

    For rectilinear arrangements, we want vertices with at most two incident
    edges per axis direction. When a vertex has multiple neighbors in a given
    direction (top/bottom/left/right), we remove all those edges to the vertex
    and reconnect them as a simple chain starting from the vertex. The resulting
    graph remains embedded rectilinearly and avoids high-degree crossings.
    """
    for v in G.nodes():
        if G.degree(v) > 2:
            # Partition neighbors by relative position w.r.t. v.
            top, bottom, left, right = [], [], [], []
            x0, y0 = v

            # Classify neighbors into top, bottom, left, or right.
            for nbr in G.neighbors(v):
                x1, y1 = nbr
                if y1 > y0:
                    top.append(nbr)
                elif y1 < y0:
                    bottom.append(nbr)
                elif x1 < x0:
                    left.append(nbr)
                elif x1 > x0:
                    right.append(nbr)

            # Sort lists by the relevant coordinate to form ordered chains.
            top.sort(key=lambda x: x[1])
            bottom.sort(key=lambda x: x[1])
            left.sort(key=lambda x: x[0])
            right.sort(key=lambda x: x[0])

            for coll in [top, bottom, left, right]:
                # If multiple neighbors in a direction, rewire them into a chain.
                if len(coll) > 1:
                    # Remove edges from v to each neighbor in this collection.
                    for nbr in coll:
                        G.remove_edge(v, nbr)

                    # Connect v to the first neighbor, then chain the rest.
                    G.add_edge(v, coll[0], is_vertical=(v[0] == coll[0][0]))
                    for i in range(len(coll) - 1):
                        G.add_edge(
                            coll[i],
                            coll[i + 1],
                            is_vertical=(coll[i][0] == coll[i + 1][0]),
                        )


def find_all_intersections(G: nx.Graph) -> None:
    """Split PSLG edges at intersections and T-junctions.

    Iterates vertical vs. horizontal edges (already sorted by helper) and:
    - If two edges cross in their interiors, inserts a new node at the
      intersection, removes the original edges, and adds four new edges.
    - If they touch at an endpoint (a T-junction), replaces the single edge
      that is intersected in its interior with two edges.
    The lists `vertical_edges` and `horizontal_edges` are updated on-the-fly to
    reflect the remaining portion of the segments as we proceed.
    """
    vertical_edges, horizontal_edges = get_vertical_and_horizontal_edges(G)

    # Iterate vertical edges (left→right) against horizontal edges (top→bottom).
    for e1 in vertical_edges:
        for i, e2 in enumerate(horizontal_edges):
            edges_adjacent = bool(set(e1) & set(e2))
            if not edges_adjacent:  # Skip if they share a vertex.
                # Build Shapely segments for geometric predicates.
                seg1 = LineString([e1[0], e1[1]])
                seg2 = LineString([e2[0], e2[1]])

                if seg1.crosses(seg2):
                    # Proper crossing: split both edges at the intersection.
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
                        logger.error("Error in crossing: (e1, e2) = %s %s", e1, e2)

                    if G.has_edge(e2[0], e2[1]):
                        G.remove_edge(e2[0], e2[1])
                    else:
                        logger.error("Error in crossing: (e1, e2) = %s %s", e1, e2)

                    # Replace e1 by its upper portion; e2 by its right portion.
                    e1 = (new_node, e1[1])
                    horizontal_edges[i] = (new_node, e2[1])

                elif seg1.touches(seg2):
                    # T-junction; split only the edge whose interior is hit.
                    inter = seg1.intersection(seg2)
                    node = (int(inter.x), int(inter.y))

                    # If an endpoint of e1 lies in the interior of e2, split e2.
                    if seg1.touches(Point(node)):
                        G.add_edge(node, e2[0], is_vertical=False)
                        G.add_edge(node, e2[1], is_vertical=False)
                        if G.has_edge(e2[0], e2[1]):
                            G.remove_edge(e2[0], e2[1])
                        else:
                            logger.error("Error in touching: (e1, e2) = %s %s", e1, e2)
                        # Keep tracking the right portion of e2.
                        horizontal_edges[i] = (node, e2[1])

                    # If an endpoint of e2 lies in the interior of e1, split e1.
                    if seg2.touches(Point(node)):
                        G.add_edge(node, e1[0], is_vertical=True)
                        G.add_edge(node, e1[1], is_vertical=True)
                        if G.has_edge(e1[0], e1[1]):
                            G.remove_edge(e1[0], e1[1])
                        else:
                            logger.error("Error in touching: (e1, e2) = %s %s", e1, e2)
                        # Keep tracking the upper portion of e1.
                        e1 = (node, e1[1])


def convert_to_PSLG(rectangles) -> nx.Graph:
    """Convert a set of rectangles into a rectilinear PSLG graph.

    - Nodes: integer lattice points with attribute `pos=(x, y)`.
    - Edges: adjacency along rectangle boundaries with attribute
      `is_vertical` to distinguish orientation.
    The graph is post-processed to resolve high-degree vertices and to split
    edges at all intersections and T-junctions.
    """
    G = nx.Graph()

    # Insert rectangle boundary edges.
    for r in rectangles:
        coords = list(r.exterior.coords)
        for i in range(len(coords) - 1):  # `exterior` closes back to start.
            v1 = (int(coords[i][0]), int(coords[i][1]))
            v2 = (int(coords[i + 1][0]), int(coords[i + 1][1]))
            G.add_node(v1, pos=coords[i])
            G.add_node(v2, pos=coords[i + 1])
            G.add_edge(v1, v2, is_vertical=(v1[0] == v2[0]))

    # Normalize topology: handle degree>2 vertices and split crossings.
    resolve_high_degree_vertices(G)
    find_all_intersections(G)

    return G


def generate_random_point_in_dcel(dcel: orthogonal_dcel.DCEL, edges, tree, args):
    """Sample a random 3D point inside the DCEL volume.

    Picks a random (x, y), locates its containing face using the vertical edge
    interval tree and ray shooting, then samples z uniformly from [0, height]
    of that face. Returns a list [x, y, z].
    """
    x = random.randint(0, args.box_width)
    y = random.randint(0, args.box_length)
    point = (x, y)
    face = dcel.face_with_point(point, edges=edges, tree=tree, is_ray_horizontal=True)
    height = face.height
    z = random.randint(0, height)
    point = [x, y, z]
    return point


def parse_args(argv=None):
    # Default parameters (used only for argparse defaults; not mutated)

    # Dimensions of the large bounding box (defaults)
    box_width = 100
    box_length = 100
    box_height = 100

    # Number of rectangles to generate (default)
    max_number_of_rectangles = 10

    # Size constraints for small rectangles (defaults)
    min_width = 1
    max_width = box_width // 2
    min_length = 1
    max_length = box_length // 2
    min_height = 1
    max_height = box_height
    parser = argparse.ArgumentParser(
        description="Generate histogram polyhedra instances."
    )

    # Output format and destination
    fmt = parser.add_mutually_exclusive_group()
    fmt.add_argument(
        "--pkl",
        dest="pkl_out",
        action="store_true",
        help="Write instances as .pkl (default: .json)",
    )
    fmt.add_argument(
        "--json",
        dest="json_out",
        action="store_true",
        help="Write instances as .json (default)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default depends on format)",
    )

    # Dimensions
    parser.add_argument(
        "--box-width",
        type=int,
        default=box_width,
        help=f"Bounding box width (default: {box_width})",
    )
    parser.add_argument(
        "--box-length",
        type=int,
        default=box_length,
        help=f"Bounding box length (default: {box_length})",
    )
    parser.add_argument(
        "--box-height",
        type=int,
        default=box_height,
        help=f"Bounding box height (default: {box_height})",
    )

    # Rectangles
    parser.add_argument(
        "--max-rectangles",
        type=int,
        default=max_number_of_rectangles,
        help=f"Max rectangles per instance (default: {max_number_of_rectangles})",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=min_width,
        help=f"Min rectangle width (default: {min_width})",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=max_width,
        help=f"Max rectangle width (default: {box_width // 2})",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=min_length,
        help=f"Min rectangle length (default: {min_length})",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=max_length,
        help=f"Max rectangle length (default: {box_length // 2})",
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=min_height,
        help=f"Min face height (default: {min_height})",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=max_height,
        help=f"Max face height (default: {int(max_height)})",
    )

    # Generation counts
    parser.add_argument(
        "--num-instances",
        type=int,
        default=999,
        help="Number of instances to generate (default: 999)",
    )
    parser.add_argument(
        "--num-s-t-pairs",
        type=int,
        default=9,
        help="Number of s/t pairs per instance (default: 9)",
    )

    # Misc
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed (default: None)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    args = parser.parse_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)

    # Configure logger
    setup_logger(args.log_level)

    # Apply seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        logger.info("Using random seed: %s", args.seed)

    # Derive format and output directory (avoid global mutation)
    use_pkl = bool(args.pkl_out) and not args.json_out
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            "../Histogram_Polyhedra_Instances_pkl"
            if use_pkl
            else "../Histogram_Polyhedra_Instances_json"
        )
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    logger.info(
        "Generating %d instances; %d s/t pairs each; format=%s",
        args.num_instances,
        args.num_s_t_pairs,
        "pkl" if use_pkl else "json",
    )

    for i in range(1, args.num_instances + 1):
        # 1) Generate a set of random rectangles.
        num_rectangles = random.randint(1, args.max_rectangles)
        rectangles = []
        while len(rectangles) < num_rectangles:
            rectangles.append(generate_random_rectangle(args))

        # Ensure a global bounding rectangle so the arrangement is closed.
        rectangles.append(box(*[0, 0], *[args.box_width, args.box_length]))

        # 2) Build PSLG and 3) convert to DCEL faces.
        G = convert_to_PSLG(rectangles)
        dcel = orthogonal_dcel.DCEL()
        dcel.compute_faces_from_graph(G)

        logger.info("Instance %d: DCEL with %d faces", i, len(dcel.faces))

        # Pre-compute vertical edge interval tree for point-location in faces.
        vertical_edges, _ = get_vertical_and_horizontal_edges(G)
        tree = create_interval_tree(edges=vertical_edges, is_vertical_tree=True)

        # 4) Assign random heights (external face has height 0).
        for face in dcel.faces:
            face.height = random.randint(args.min_height, args.max_height)
            if face.is_external:
                face.height = 0

        # 5) Sample s/t point pairs and serialize instances.
        for j in range(1, args.num_s_t_pairs + 1):
            s = generate_random_point_in_dcel(
                dcel=dcel, edges=vertical_edges, tree=tree, args=args
            )
            t = generate_random_point_in_dcel(
                dcel=dcel, edges=vertical_edges, tree=tree, args=args
            )
            dcel.s = s
            dcel.t = t
            if use_pkl:
                filename = os.path.join(output_dir, f"instance_{i:03}_{j:03}.pkl")
                success = dcel.save_to_pickle(filename)
                if not success:
                    logger.error("Pickling error %s", filename)
            else:
                filename = os.path.join(output_dir, f"instance_{i:03}_{j:03}.json")
                dcel.save_to_json(filename)
            logger.debug("Saved %s", filename)


if __name__ == "__main__":
    main()

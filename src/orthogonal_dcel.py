import math
import networkx as nx
from typing import List, Tuple, Optional, Iterator
from functools import cached_property
from orthogonal_ray_shooting import (
    create_interval_tree,
    ray_shooting,
    get_vertical_and_horizontal_edges,
)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pickle
from orthogonal_ray_shooting import ray_shooting
import json
from itertools import combinations


class Vertex:  # the Vertex class contains 3 attributes: .x, .y, .incident_half_edges
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.incident_half_edges = []  # Half-edges with this vertex as origin

    def __eq__(self, other) -> bool:  # check if other is equal to self
        if not isinstance(other, Vertex):
            return False
        return self.x == other.x and self.y == other.y

    def __hash__(self):  # hash thingy for the equal to work
        return hash((self.x, self.y))

    def __repr__(self):  # how this class should be printed
        return f"Vertex({self.x}, {self.y})"

    @property
    def coords(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def sort_incident(
        self,
    ) -> None:  # sorts the incident edges in counter-clockwise order
        self.incident_half_edges.sort(key=lambda h: h.angle, reverse=True)


class HalfEdge:  # the half-edge class contains the following attributes: .origin, .destination, .twin, .next, .prev, .face; note that half edges are, therefore, directed
    def __init__(self, origin=None, destination=None):
        self.origin: Optional[Vertex] = origin  # Vertex where this half-edge starts
        self.destination: Optional[Vertex] = (
            destination  # Vertex where this half-edge ends
        )
        self.twin: Optional[HalfEdge] = None  # Twin half-edge
        self.next: Optional[HalfEdge] = None  # Next half-edge around the face
        self.prev: Optional[HalfEdge] = None  # Previous half-edge around the face
        self.face: Optional[Face] = None  # Face this half-edge borders

    def __eq__(self, other):
        if not isinstance(other, HalfEdge):
            return False
        return (self.origin == other.origin) and (self.destination == other.destination)

    def __hash__(self):
        return hash((self.origin, self.destination))

    def __repr__(self):
        return f"({self.origin.coords if self.origin else None} -> {self.destination.coords if self.destination else None})"

    @cached_property
    def angle(
        self,
    ) -> float:  # this computes the angle between this half-edge and the x-axis
        dx = self.destination.x - self.origin.x
        dy = self.destination.y - self.origin.y
        angle = math.atan2(dy, dx)
        return angle if angle >= 0 else angle + 2 * math.pi

    def _is_point_to_left(
        self, point
    ) -> (
        bool
    ):  # decide if the the given point (a tuple) is to the left of this half edge
        # Vector from origin to destination
        u = self.origin
        v = self.destination

        u = u.coords
        v = v.coords

        edge_vector = (v[0] - u[0], v[1] - u[1])

        # Vector from origin to the point
        point_vector = (point[0] - u[0], point[1] - u[1])

        # Cross product: positive if point is to the left
        cross_product = (
            edge_vector[0] * point_vector[1] - edge_vector[1] * point_vector[0]
        )

        return cross_product > 0


class Face:  # the Face class has three attributes: .start_half_edge, .inner_components_half_edges, .is_external
    def __init__(self):
        self.start_half_edge: Optional[HalfEdge] = (
            None  # A half-edge bounding the outer face cycle -- note that this remains None for the external face
        )
        self.inner_components_half_edges = []  # Half edges from each of the inner cycles of the face
        self.is_external: Optional[bool] = (
            None  # True if the face is external; False otherwise
        )
        self.height: Optional[int] = None  # Height of the face

    def __repr__(self):
        return f"Face(outer={self.start_half_edge}, inner = {self.inner_components_half_edges}, is external = {self.is_external})"

    def _vertices_in_cycle(
        self, start_half_edge
    ) -> Iterator[
        Vertex
    ]:  # yields the vertices around the cycle starting at the given edge
        if not start_half_edge:
            return

        start = start_half_edge
        current = start
        while True:
            yield current.origin
            current = current.next
            if current == start:
                break

    def _edges_in_cycle(
        self, start_half_edge
    ) -> Iterator[
        HalfEdge
    ]:  # yields the edges around the cycle starting at the given edge
        if not start_half_edge:
            return

        start = start_half_edge
        current = start
        while True:
            yield current
            current = current.next
            if current == start:
                break

    def outer_vertices(
        self,
    ) -> Iterator[Vertex]:  # yields the vertices that bound this face
        if not self.start_half_edge:
            return

        yield from self._vertices_in_cycle(start_half_edge=self.start_half_edge)

    def inner_vertices(
        self,
    ) -> Iterator[List[Vertex]]:  # vertices of the holes inside the face
        if not self.inner_components_half_edges:
            return

        for he in self.inner_components_half_edges:
            yield list(self._vertices_in_cycle(he))

    def outer_edges(
        self,
    ) -> Iterator[HalfEdge]:  # yields the edges that bound this face
        if not self.start_half_edge:
            return

        yield from self._edges_in_cycle(start_half_edge=self.start_half_edge)

    def inner_edges(
        self,
    ) -> Iterator[List[HalfEdge]]:  # edges of the holes inside the face
        if not self.inner_components_half_edges:
            return

        for he in self.inner_components_half_edges:
            yield list(self._edges_in_cycle(he))

    def is_vertex_outer(self, vertex: Vertex) -> bool:
        return vertex in list(self.outer_vertices)


class DCEL:  # The DCEL class has .vertices, .half_edges, and .faces as its attributes
    def __init__(self):
        self.vertices: List[Vertex] = []
        self.half_edges: List[HalfEdge] = []
        self.faces: List[Face] = []
        self.s: List[int] = []
        self.t: List[int] = []

    def __repr__(self):
        return f"DCEL(vertices={self.vertices};\n\n edges={self.half_edges};\n\n faces={self.faces})"

    def _add_vertex(self, x, y) -> Vertex:  # adds a vertex to the DCEL
        v = Vertex(x, y)
        self.vertices.append(v)
        return v

    def _add_edge_pair(
        self, v1, v2
    ) -> Tuple[
        HalfEdge
    ]:  # adds the half edge v1v2 and the half edge v2v1 to the DCEL; note that v1 and v2 are in the Vertex class
        he1 = HalfEdge()
        he2 = HalfEdge()
        he1.origin = v1
        he1.destination = v2
        he2.origin = v2
        he2.destination = v1
        he1.twin = he2
        he2.twin = he1
        self.half_edges.extend([he1, he2])

        v1.incident_half_edges.append(he1)
        v2.incident_half_edges.append(he2)
        return he1, he2

    def _set_next_prev_pointers(
        self,
    ) -> None:  # this sets the prev and next pointers of half edges
        for vertex in self.vertices:  # we do so by checking the incident half edges of each vertex. These are already sorted around the vertex; half edge i + 1 is the next edge of the twin of half edge i
            incident = vertex.incident_half_edges
            n = len(incident)

            for i in range(n):
                current_edge = incident[i]
                next_edge = incident[(i + 1) % n]

                # The next edge of current_edge.twin is next_edge
                current_edge.twin.next = next_edge
                next_edge.prev = current_edge.twin

    def _build_dcel_vertices_and_half_edges_from_graph(
        self, G
    ) -> None:  # given a networkx graph, we compute the vertices and half edges of our DCEL first
        vertex_map = {}  # Create vertex map; this will help with the half edges

        for node in G.nodes():  # Create vertices
            x, y = node
            vertex = self._add_vertex(x, y)
            vertex_map[node] = vertex

        for u, v in G.edges():  # Create half-edges
            v1 = vertex_map[u]
            v2 = vertex_map[v]
            self._add_edge_pair(v1, v2)

        for vertex in self.vertices:  # Sort incident edges around each vertex
            vertex.sort_incident()

        self._set_next_prev_pointers()  # Set next and prev pointers

    def _find_all_cycles(
        self,
    ) -> List[
        List[HalfEdge]
    ]:  # we now compute cycles in the DCEL; each cycle is represented by a list of half edges
        unprocessed = set(self.half_edges)
        cycles = []

        while unprocessed:
            start_edge = (
                unprocessed.pop()
            )  # Pick any unprocessed half-edge to start a cycle
            cycle = []
            current = start_edge

            while True:  # Trace cycle starting from this half-edge by moving to the next pointer iteratively until you are back at the start edge
                unprocessed.discard(current)
                cycle.append(current)
                current = current.next
                if current == start_edge:
                    break

            if len(cycle) >= 3:  # Sanity check for a valid cycle
                cycles.append(cycle)

        return cycles

    def _left_bottom_edge(
        self, cycle
    ) -> (
        HalfEdge
    ):  # Returns the edge which has the left-bottom vertex of the cycle as its origin
        left_bottom_edge = cycle[0]
        left_bottom_vertex = left_bottom_edge.origin

        for edge in cycle:
            v = edge.origin
            if (
                v.x < left_bottom_vertex.x
            ):  # if it is strictly to the left, update our values
                left_bottom_vertex = v
                left_bottom_edge = edge
            elif (
                v.x == left_bottom_vertex.x
            ):  # if its equal, check if the y-coordinate is lower
                if v.y < left_bottom_vertex.y:
                    left_bottom_vertex = v
                    left_bottom_edge = edge

        return left_bottom_edge

    def _is_inner_cycle(
        self, cycle
    ) -> bool:  # this check if the cycle is an inner circle or an outer circle; this is simply by checking what angle the edges of the cycle which has the left-bottom-vertex as its origin and destination make
        he1 = self._left_bottom_edge(cycle)
        he2 = he1.prev

        he1_vector = (
            (he1.origin).x - (he1.destination).x,
            (he1.origin).y - (he1.destination).y,
        )
        he2_vector = (
            (he2.origin).x - (he2.destination).x,
            (he2.origin).y - (he2.destination).y,
        )

        cross = he1_vector[0] * he2_vector[1] - he2_vector[0] * he1_vector[1]
        if cross > 0:
            return False  # angle from he1 to he2 is < 180°
        elif cross < 0:
            return True  # angle from he1 to he2 is > 180°
        else:
            return None

    def _determine_cycle_with_edge(
        self, cycles, edge, point
    ) -> int:  # this gives the index of the cycle which contains a given edge and has the point to its left
        v1 = Vertex(edge[0][0], edge[0][1])
        v2 = Vertex(edge[1][0], edge[1][1])

        he = HalfEdge()
        he.origin = v1
        he.destination = v2
        if not he._is_point_to_left(
            point
        ):  # if this direction does not have the point to its left, then flip it around
            he.origin = v2
            he.destination = v1
        # return next((j for (j, cycle) in enumerate(cycles) if next()))
        for j, cycle in enumerate(cycles):
            for edge in cycle:
                if he == edge:
                    return j

    def _compute_connectivity_graph(
        self, cycles, vertical_edges, vertical_edges_tree
    ):  # this constructs a graph on the cycles to compute which inner cycle contains which collection of outer cycles
        C = nx.Graph()

        for i, cycle in enumerate(cycles):  # the nodes of the graph are cycles
            C.add_node(i, cycle=cycle)

        for i, cycle in enumerate(cycles):
            if not self._is_inner_cycle(
                cycle
            ):  # if this is an outer cycle, compute the left bottom vertex of it
                left_bottom_vertex = (self._left_bottom_edge(cycle)).origin
                left_bottom_vertex = (left_bottom_vertex.x, left_bottom_vertex.y)
                if (
                    left_bottom_vertex[0] > 0
                ):  # NOTE: we assume here that the external face is the only one which has a non-positive x-coordinate
                    _, bisected_edge_index = ray_shooting(
                        point=left_bottom_vertex,
                        tree=vertical_edges_tree,
                        is_left=1,
                        is_down=0,
                        is_ray_horizontal=1,
                    )  # shoot a ray to the left
                    bisected_edge = vertical_edges[
                        bisected_edge_index
                    ]  # this is the edge it intersects
                    j = self._determine_cycle_with_edge(
                        cycles, bisected_edge, left_bottom_vertex
                    )  # compute which cycle it belongs to
                    C.add_edge(i, j)  # add this edge
        return C

    def _construct_faces_from_cycles(
        self, cycles, vertical_edges
    ):  # we finally construct the faces of our DCEL
        vertical_edges_tree = create_interval_tree(
            vertical_edges, is_vertical_tree=True
        )

        H = self._compute_connectivity_graph(
            cycles, vertical_edges, vertical_edges_tree
        )  # Build containment graph H using our ray shooting method
        components = list(nx.connected_components(H))  # Find connected components in H

        for component in components:  # Create faces from connected components
            component_cycles = [
                cycles[i] for i in component
            ]  # Get cycles in this component
            inner_cycles_in_component = []  # Separate inner and outer cycles in this component
            outer_cycles_in_component = []
            for cycle in component_cycles:
                if self._is_inner_cycle(cycle):
                    inner_cycles_in_component.append(cycle)
                else:
                    outer_cycles_in_component.append(cycle)

            face = Face()  # Create face
            face.inner_components_half_edges = []

            if (
                len(inner_cycles_in_component) == 1
            ):  # The inner cycle is the outer boundary of this face; if there is none, then it is the external face
                outer_boundary = inner_cycles_in_component[0]
                face.start_half_edge = outer_boundary[0]
                face.is_external = False

                for edge in outer_boundary:  # Set face pointer for outer boundary edges
                    edge.face = face

                for outer_cycle in outer_cycles_in_component:  # Outer cycles in component become holes; so you put pointers to them in the inner_component_half_edges attribute
                    face.inner_components_half_edges.append(outer_cycle[0])
                    for edge in outer_cycle:
                        edge.face = face

            elif (
                len(inner_cycles_in_component) == 0
                and len(outer_cycles_in_component) > 0
            ):
                face.is_external = True  # This is the external face - outer cycles are holes in external face
                face.start_half_edge = None  # External face has no outer boundary

                for outer_cycle in outer_cycles_in_component:
                    face.inner_components_half_edges.append(outer_cycle[0])
                    for edge in outer_cycle:
                        edge.face = face

            self.faces.append(face)  # add this to the faces of self

    def _generate_3d_polygons(
        self,
    ):  # generate all the polyhedra that make up the histogram
        polygons_3d = []

        for face in self.faces:  # iterate over faces
            if face.is_external:
                continue  # Skip external face

            height = getattr(face, "height", 0)
            if height <= 0:
                continue  # Skip faces with no height

            face_vertices_2d = list(
                face.outer_vertices()
            )  # Get 2D vertices of the face boundary using face's vertices method

            if len(face_vertices_2d) < 3:
                continue  # Skip degenerate faces

            bottom_vertices_3d = [
                (v.x, v.y, 0) for v in face_vertices_2d
            ]  # Bottom face (z = 0)
            polygons_3d.append(
                {
                    "vertices": bottom_vertices_3d,
                    "type": "bottom",
                    "face": face,
                    "height": 0,
                }
            )

            top_vertices_3d = [
                (v.x, v.y, height) for v in face_vertices_2d
            ]  # Top face (z = height)
            polygons_3d.append(
                {
                    "vertices": top_vertices_3d,
                    "type": "top",
                    "face": face,
                    "height": height,
                }
            )

            n = len(face_vertices_2d)  # Side faces connecting bottom and top
            for i in range(n):
                v1 = face_vertices_2d[i]
                v2 = face_vertices_2d[(i + 1) % n]

                # Create rectangular side face
                side_face = [
                    (v1.x, v1.y, 0),  # bottom-left
                    (v2.x, v2.y, 0),  # bottom-right
                    (v2.x, v2.y, height),  # top-right
                    (v1.x, v1.y, height),  # top-left
                ]

                polygons_3d.append(
                    {
                        "vertices": side_face,
                        "type": "side",
                        "face": face,
                        "height": height,
                    }
                )

            # Handle holes in the face using inner_components_half_edges
            for hole_vertices_2d in face.inner_vertices():
                # Bottom hole (subtract from bottom face)
                bottom_hole_3d = [(v.x, v.y, 0) for v in reversed(hole_vertices_2d)]
                polygons_3d.append(
                    {
                        "vertices": bottom_hole_3d,
                        "type": "bottom_hole",
                        "face": face,
                        "height": 0,
                    }
                )

                # Top hole (subtract from top face)
                top_hole_3d = [(v.x, v.y, height) for v in hole_vertices_2d]
                polygons_3d.append(
                    {
                        "vertices": top_hole_3d,
                        "type": "top_hole",
                        "face": face,
                        "height": height,
                    }
                )

                # Inner side faces (walls of the hole)
                n_hole = len(hole_vertices_2d)
                for i in range(n_hole):
                    v1 = hole_vertices_2d[i]
                    v2 = hole_vertices_2d[(i + 1) % n_hole]

                    # Create rectangular inner side face (reversed orientation)
                    inner_side_face = [
                        (v1.x, v1.y, height),  # top-left
                        (v2.x, v2.y, height),  # top-right
                        (v2.x, v2.y, 0),  # bottom-right
                        (v1.x, v1.y, 0),  # bottom-left
                    ]

                    polygons_3d.append(
                        {
                            "vertices": inner_side_face,
                            "type": "inner_side",
                            "face": face,
                            "height": height,
                        }
                    )

        return polygons_3d

    def _get_max_height(self):  # Get maximum height among all faces
        max_height = 0
        for face in self.faces:
            if hasattr(face, "height"):
                max_height = max(max_height, face.height)
        return max_height if max_height > 0 else 1

    def _set_axis_properties_3d(
        self, ax
    ):  # Set 3D axis properties for better visualization
        # Get bounds from DCEL
        all_x = []
        all_y = []
        all_z = []

        for vertex in self.vertices:
            all_x.append(vertex.x)
            all_y.append(vertex.y)

        for face in self.faces:
            if hasattr(face, "height"):
                all_z.append(face.height)

        if all_x and all_y:
            x_range = max(all_x) - min(all_x)
            y_range = max(all_y) - min(all_y)
            z_range = max(all_z) - min(all_z) if all_z else 1

            # Set equal aspect ratio
            max_range = max(x_range, y_range, z_range)

            x_center = (max(all_x) + min(all_x)) / 2
            y_center = (max(all_y) + min(all_y)) / 2
            z_center = (max(all_z) + min(all_z)) / 2 if all_z else 0

            ax.set_xlim([x_center - max_range / 2, x_center + max_range / 2])
            ax.set_ylim([y_center - max_range / 2, y_center + max_range / 2])
            ax.set_zlim([0, max(all_z) if all_z else 1])

    def face_with_point(
        self, point, edges, tree, is_ray_horizontal
    ) -> Face:  # return the face in the DCEL that the point is in
        if is_ray_horizontal:
            is_left = True
            is_down = False
        else:
            is_left = False
            is_down = True

        _, bisected_edge_index = ray_shooting(
            point=point,
            tree=tree,
            is_left=is_left,
            is_down=is_down,
            is_ray_horizontal=is_ray_horizontal,
        )  # shoot a ray to the left or down
        bisected_edge = edges[
            bisected_edge_index
        ]  # compute the edge that is bisected edge

        v1 = Vertex(bisected_edge[0][0], bisected_edge[0][1])
        v2 = Vertex(bisected_edge[1][0], bisected_edge[1][1])

        he = HalfEdge()
        he.origin = v1
        he.destination = v2

        if not he._is_point_to_left(
            point
        ):  # if this direction does not have the point to its left, then flip it around
            he.origin = v2
            he.destination = v1

        for he2 in (
            self.half_edges
        ):  # find the half edge which is equal to he and return the face
            if he2 == he:
                return he2.face

    def compute_faces_from_graph(self, G) -> "DCEL":
        self._build_dcel_vertices_and_half_edges_from_graph(
            G
        )  # Step 1: Build DCEL structure from graph
        cycles = self._find_all_cycles()  # Step 2: Find all cycles in the DCEL

        vertical_edges, _ = get_vertical_and_horizontal_edges(G)

        self._construct_faces_from_cycles(
            cycles, vertical_edges
        )  # Step 3: Determine face structure from cycles

        return self

    def convert_to_graph(self):  # Convert DCEL to networkx graph
        G = nx.Graph()

        for vertex in self.vertices:
            G.add_node(vertex.coords, pos=vertex.coords)

        for he in self.half_edges:
            v1 = (he.origin).coords
            v2 = (he.destination).coords
            G.add_edge(v1, v2, is_vertical=(v1[0] == v2[0]))

        return G

    def get_graph_for_GCS(
        self, edges, tree
    ):  # given the decomposed DCEL, This returns the GCS graph
        unprocessed = set(self.half_edges)
        H = nx.Graph()

        def he_to_edge(
            he: HalfEdge,
        ):  # Convert half-edge to ordered edge (left to right)
            p1 = he.origin.coords
            p2 = he.destination.coords

            if p1[0] < p2[0] or (
                p1[0] == p2[0] and p1[1] < p2[1]
            ):  # Order from left to right (smaller x first, then smaller y if x equal)
                return (p1, p2)
            else:
                return (p2, p1)

        def connect_point_to_face_edges(
            point_vertex, edges, tree
        ) -> Face:  # Connect a point vertex to all edges bounding the given face
            point = (point_vertex[0], point_vertex[1])

            face = self.face_with_point(
                point=point, edges=edges, tree=tree, is_ray_horizontal=True
            )  # compute the face which has the point
            if face:
                face_edges = list(face.outer_edges())
                for he in (
                    face_edges
                ):  # add an edge between the point and every edge bounding the face
                    edge_vertex = he_to_edge(he)
                    H.add_edge(point_vertex, edge_vertex)
            return face

        while unprocessed:  # Vertices of H are the edges of the graph (not half edges)
            he = unprocessed.pop()  # remove the half edge and its twin
            unprocessed.discard(he.twin)

            min_height = min(
                he.face.height, he.twin.face.height
            )  # the intersection between the two faces will be the rectangle whose height is the minimum of the heights of the faces that are in either side of this edge
            diagonals = (he.origin.coords + (0,), he.destination.coords + (min_height,))

            edge_vertex = he_to_edge(he)  # Use ordered edge as vertex name
            H.add_node(
                edge_vertex, diagonals=diagonals
            )  # the "diagonals" field will hold the diagonal vertices

        for face in self.faces:  # Add edges based on faces -- a face induces a clique; we only use the outer faces since the final decomposition has no inner components
            face_half_edges = list(face.outer_edges())

            for he1, he2 in combinations(face_half_edges, 2):
                edge1 = he_to_edge(he1)
                edge2 = he_to_edge(he2)
                H.add_edge(edge1, edge2)

        H.add_node(tuple(self.s), is_source=True)
        H.add_node(tuple(self.t), is_destination=True)

        face_s = connect_point_to_face_edges(
            tuple(self.s), edges, tree
        )  # this is the face that s is in
        face_t = connect_point_to_face_edges(
            tuple(self.t), edges, tree
        )  # this is the face that t is in

        if (
            face_s == face_t
        ):  # if they are in the same face, then add an edge between them
            H.add_edge(tuple(self.s), tuple(self.t))
        return H

    def save_to_pickle(self, filename) -> bool:  # Save DCEL to pickle file
        try:
            with open(filename, "wb") as f:
                pickle.dump(self, f)
            return True
        except RecursionError:
            return False

    @classmethod
    def load_from_pickle(cls, filename) -> "DCEL":  # Load DCEL from pickle file
        try:
            with open(filename, "rb") as f:
                dcel = pickle.load(f)
            return dcel
        except FileNotFoundError:
            return FileNotFoundError

    def save_to_json(self, filename, G=None):
        if G is None:
            G = self.convert_to_graph()

        half_edge_height_dict = {}

        for he in self.half_edges:
            he_coords = (he.origin.coords, he.destination.coords)
            half_edge_height_dict[he_coords] = he.face.height

        graph_data = nx.node_link_data(G)
        graph_data["half_edge_height_dict"] = {
            json.dumps(edge): value for edge, value in half_edge_height_dict.items()
        }
        graph_data["source_and_destination"] = {"source": self.s, "destination": self.t}

        with open(filename, "w") as f:
            json.dump(graph_data, f, indent=2)

    @classmethod
    def load_from_json(cls, filename) -> "DCEL":  # load
        try:
            with open(filename, "r") as f:
                data = json.load(f)

            half_edge_height_dict = {}
            for k, v in data.pop("half_edge_height_dict").items():
                edge_list = json.loads(k)
                half_edge_height_dict[(tuple(edge_list[0]), tuple(edge_list[1]))] = v

            s = data["source_and_destination"]["source"]
            t = data["source_and_destination"]["destination"]

            G = nx.node_link_graph(data)

            dcel = DCEL()
            dcel.compute_faces_from_graph(G)

            for face in dcel.faces:
                if face.is_external:
                    face.height = 0

                else:
                    he = face.start_half_edge
                    he_coords = (he.origin.coords, he.destination.coords)
                    height = half_edge_height_dict[he_coords]
                    face.height = height

            dcel.s = s
            dcel.t = t

            return dcel

        except FileNotFoundError:
            return FileNotFoundError

    def plot_histogram_polyhedron(
        self, alpha=0.15, show_wireframe=True, face_colors=None
    ):
        """
        Plot the 3D histogram polyhedron from DCEL with face heights

        Args:
            alpha: Transparency of faces (0-1)
            show_wireframe: Whether to show edge wireframe
            face_colors: Dict mapping face to color, or None for auto-coloring
        """

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")

        polygons_3d = (
            self._generate_3d_polygons()
        )  # Generate all 3D polygons for the polyhedron

        # Group polygons by face and assign consistent colors
        face_color_map = {}
        all_faces = []
        colors = []

        for face in self.faces:  # First pass: determine color for each DCEL face
            if face.is_external or not hasattr(face, "height") or face.height <= 0:
                continue

            if face_colors and face in face_colors:
                face_color_map[face] = face_colors[face]
            else:  # Auto-color based on height
                height = face.height
                color_intensity = min(1.0, height / self._get_max_height())
                face_color_map[face] = plt.cm.viridis(color_intensity)

        for poly_info in (
            polygons_3d
        ):  # Second pass: assign colors to all polygons based on their DCEL face
            poly = poly_info["vertices"]
            face = poly_info.get("face", None)

            all_faces.append(poly)

            if face and face in face_color_map:
                colors.append(face_color_map[face])
            else:  # Default color for polygons without associated face
                colors.append("lightgray")

        if all_faces:  # Plot all polygons with consistent face coloring
            poly_collection = Poly3DCollection(
                all_faces,
                alpha=alpha,
                facecolors=colors,
                edgecolors="black" if show_wireframe else None,
            )
            ax.add_collection3d(poly_collection)

        ax.scatter(self.s[0], self.s[1], self.s[2], color="blue", s=20, marker="s")
        ax.scatter(self.t[0], self.t[1], self.t[2], color="darkgreen", s=20, marker="s")

        self._set_axis_properties_3d(ax)  # Set axis properties

        # Add labels and title
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Height")
        ax.set_title("3D Histogram Polyhedron from DCEL")
        ax.view_init(elev=30, azim=-60)

        # Add color bar for height mapping
        if face_color_map:
            heights = [face.height for face in face_color_map.keys()]
            if heights:
                sm = plt.cm.ScalarMappable(
                    cmap=plt.cm.viridis,
                    norm=plt.Normalize(vmin=min(heights), vmax=max(heights)),
                )
                sm.set_array([])
                plt.colorbar(sm, ax=ax, label="Face Height", shrink=0.5)

        plt.tight_layout()
        # plt.show()

        return fig, ax

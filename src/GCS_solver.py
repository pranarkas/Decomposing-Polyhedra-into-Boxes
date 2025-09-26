import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pydrake.geometry.optimization import GraphOfConvexSets, Point, HPolyhedron, GraphOfConvexSetsOptions
import re
from pydrake.solvers import MosekSolver, GurobiSolver
import logging
from format_logger import setup_logger

# Setup logging
setup_logger(level="INFO")
logger = logging.getLogger(__name__)

def convert_node_to_string(node: nx.Graph.nodes):
    string = f"{node}"
    return string

def convert_to_pydrake_gcs_format(H: nx.Graph):
    """
    Convert your graph to PyDrake's Graph of Convex Sets format.
    
    Args:
        nodes: Dictionary where keys are node names and values contain node attributes
               For source/target: {'is_source': True, 'is_destination': True}
               For regular nodes: {'diagonal': [(x1,y1,z1), (x2,y2,z2)]}
        edges: List of tuples (from_node, to_node) or list of edge objects
        source_node: Name of the source node
        target_node: Name of the target node
    
    Returns:
        gcs: Graph of Convex Sets object
        vertex_map: Dictionary mapping node names to GCS vertices
    """
    
    # Create the Graph of Convex Sets
    gcs = GraphOfConvexSets()
    vertex_map = {}
    
    D = H.to_directed()

    for node, attrs in D.nodes(data=True):
        diagonal = attrs.get('diagonals', None)
        if diagonal is None: #if it has no diagonal, then it is a point
            convex_set = Point(np.array(node))
            #logger.info(f"Node {node} is a point.")
        else: # it is a box defined by its diagonal
            min_point = np.array(diagonal[0])
            max_point = np.array(diagonal[1])  
            #logger.info(f"Node {node} has diagonals with min_point: {min_point} and max_point: {max_point}")
            convex_set = HPolyhedron.MakeBox(lb = min_point, ub = max_point)

        
        node = convert_node_to_string(node) # Convert node to string for consistent naming
        vertex = gcs.AddVertex(convex_set, name=node) # Add vertex to GCS
        vertex_map[node] = vertex # Map node name to vertex
    
    # Add edges to the GCS
    for edge in D.edges():
        from_node, to_node = edge
        
        from_node = convert_node_to_string(from_node)
        to_node = convert_node_to_string(to_node)

        from_vertex = vertex_map[from_node]
        to_vertex = vertex_map[to_node]
        
        # Add edge with Euclidean distance cost (L2 norm)
        edge = gcs.AddEdge(from_vertex, to_vertex)
        xu = edge.xu()   # decision var: point in source set
        xv = edge.xv()   # decision var: point in target set
        edge.AddCost(np.sqrt((xu - xv).dot(xu - xv)))   # Euclidean distance
    
    return gcs, vertex_map

def get_pos(name):
    """
    Parse a vertex name string into a 2D position (x,y).
    - "(x, y, z)" → (x,y)
    - "(x1, y1, z1), (x2, y2, z2)" → midpoint projected to (x,y)
    """
    nums = list(map(float, re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", name)))
    if len(nums) == 3:
        coords = np.array(nums[:2])         # drop z
    elif len(nums) == 4:
        p1, p2 = np.array(nums[:2]), np.array(nums[2:])
        coords = ((p1 + p2) / 2.0)[:2]      # midpoint, drop z
    else:
        raise ValueError(f"Unrecognized vertex name format: {name}")
    return coords


def draw_GCS_with_flows(gcs: GraphOfConvexSets, source, target, flows):
    """
    Visualize the Graph of Convex Sets with different colors for source, target, and edges with flow.
    Args:
        gcs: Graph of Convex Sets
        source: Source node name
        target: Target node name
        flows: Dictionary mapping (from_node, to_node) to flow value
    """
    G = nx.DiGraph()
    for v in gcs.Vertices():
        G.add_node(v.name())
    for e in gcs.Edges():
        G.add_edge(e.u().name(), e.v().name())

    pos = {v.name(): get_pos(v.name()) for v in gcs.Vertices()}

    plt.figure(figsize=(8, 6))

    # Draw nodes with custom colors
    node_colors = []
    for v in gcs.Vertices():
        if v.name() == convert_node_to_string(source):
            node_colors.append("darkgreen")
        elif v.name() == f"{target}":
            node_colors.append("red")
        else:
            node_colors.append("lightblue")

    edge_color = []
    edge_width = []
    for e in gcs.Edges():
        if flows.get((e.u().name(), e.v().name()), 0) > 1e-3:
            edge_color.append('blue')
            edge_width.append(2.0)
        else:
            edge_color.append('gray')
            edge_width.append(0.5)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=20)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, width=edge_width, arrows=True, arrowsize=10)
    

    plt.axis("equal")
    plt.axis("off")
    #plt.show()

def find_shortest_path(gcs: GraphOfConvexSets, vertex_map, source_node, target_node, 
                       solver_options=None):
    """
    Solve the shortest path problem using PyDrake's optimization.
    
    Args:
        gcs: Graph of Convex Sets
        vertex_map: Dictionary mapping node names to vertices
        source_node: Name of source node
        target_node: Name of target node
        solver_options: Optional solver options
    
    Returns:
        result: Optimization result
        path_points: List of 3D points along the optimal path
    """
    # Solve the shortest path problem
    options = GraphOfConvexSetsOptions()
    #options.convex_relaxation = True 
    options.solver = MosekSolver() 

    source_node = convert_node_to_string(source_node)
    target_node = convert_node_to_string(target_node)

    logger.info(f"Finding shortest path from {source_node} to {target_node}...")
    result = gcs.SolveShortestPath(
        source=vertex_map[source_node], 
        target=vertex_map[target_node],
        options= options
    )
    logger.info(f"Solve result: {result.get_solution_result()}")
    # Extract flows on edges
    flows = {}
    for e in gcs.Edges():
        flow_val = result.GetSolution(e.phi())
        if flow_val > 1e-3:
            logger.info( f"Edge from {e.u().name()} to {e.v().name()} has flow {flow_val}" )
        flows[(e.u().name(), e.v().name())] = flow_val

    path = None
    path_points = None
    if result.is_success():
        path = gcs.GetSolutionPath(source=vertex_map[source_node], 
            target=vertex_map[target_node],result=result)
    
    path_points = [e.u().GetSolution(result=result) for e in path]
    e = path[-1]
    path_points.append(e.v().GetSolution(result=result))

    return result, path_points, flows
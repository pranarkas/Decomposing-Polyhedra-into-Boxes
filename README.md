# Introduction

This repository contains four main components to generate and decompose histogram polyhedra:

1. **`orthogonal_ray_shooting.py`** - Efficient ray shooting queries in orthogonal line arrangements
2. **`orthogonal_dcel.py`** - Doubly Connected Edge List (DCEL) data structure for planar subdivisions
3. **`generate_histogram_polyhedra.py`** - Random 3D histogram polyhedra generation from overlapping rectangles
4. **`decompose_histogram_polyhedra.py`** - A 4-approximation algorithm to partition 3D histogram polyhedra into axes-aligned rectangular boxes

# Overview of the Components

## Orthogonal Ray Shooting (`orthogonal_ray_shooting.py`)

Provides efficient ray shooting operations for orthogonal line arrangements using interval trees. Key capabilities:
- Extract and sort vertical/horizontal edges from graphs
- Build interval trees for fast range queries
- Perform ray shooting in 4 directions (left, right, up, down)
- Find intersection points and bisected edges

## Orthogonal DCEL (`orthogonal_dcel.py`)

Implements a complete Doubly Connected Edge List data structure optimized for orthogonal planar subdivisions:
- **Vertex class**: Stores coordinates and incident half-edges with counter-clockwise ordering
- **HalfEdge class**: Directed edges with twin pointers and face connectivity
- **Face class**: Bounded regions with outer boundaries and inner holes
- **DCEL class**: Complete planar subdivision with topological consistency

## Histogram Polyhedra Generator (`generate_histogram_polyhedra.py`)

Creates geometric instances suitable for orthogonal decomposition algorithms by:
1. Generating random 2D rectangles within a bounding box
2. Converting overlapping rectangles into a Planar Straight Line Graph (PSLG)
3. Building a DCEL representation with face information
4. Assigning random heights to create 3D histogram polyhedra
5. Saving instances as pickled files for later use to ``Histogram_Polyhedra_Instances``  

## Decompose Histogram Polyhedra (`decompose_histogram_polyhedra.py`)

Reads and visualizes a pickled histogram polyhedron and decomposes it into boxes by:

1. Reading and outputting random pickle file from ``Histogram_Polyhedra_Instances``
2. Converting the DCEL into a PSLG
3. Optimally partitioning the PSLG into rectangles
4. Extruding the rectangles to their required heights
5. Illustrating this partition of the polyhedron

# Running Time Analysis

- **DCEL Construction**: $O(n + m \log{m})$ for $n$ vertices and $m$ edges using ray shooting for face determination
- **Ray Shooting Queries**: $O(\log n)$ per query using interval trees; constructing an interval tree takes $O(n\log n)$ time
- **Point Location**: $O(\log n)$ per query
- **Face Traversal**: $O(k)$ for boundary with $k$ edges
- **3D Visualization**: $O(f)$ for $f$ faces with holes handled efficiently

The implementation is optimized for orthogonal arrangements where all edges are axis-aligned, enabling specialized algorithms and data structures for improved performance.

# Dependencies

```bash
pip install shapely networkx intervaltree matplotlib
```

**Custom modules required:**
- All three modules are interdependent and should be in the same directory

# Ray Shooting Module (`orthogonal_ray_shooting.py`)

## Functions

### `get_vertical_and_horizontal_edges(G)`
Extracts and sorts edges from a NetworkX graph containing orthogonal line segments.

**Parameters:**
- `G`: NetworkX graph with edges having `is_vertical` boolean attribute

**Returns:**
- `[vertical_edges, horizontal_edges]`: Two lists of edge tuples
  - Vertical edges sorted left-to-right by x-coordinate
  - Horizontal edges sorted bottom-to-top by y-coordinate

**Edge format:** Each edge is `((x1,y1), (x2,y2))` with consistent orientation:
- Vertical edges: lower vertex first (sorted by y)
- Horizontal edges: left vertex first (sorted by x)

### `create_interval_tree(edges, is_vertical_tree)`
Builds an interval tree for efficient range queries on edge collections.

**Parameters:**
- `edges`: List of edge tuples from `get_vertical_and_horizontal_edges()`
- `is_vertical_tree`: Boolean
  - `True`: Build tree for vertical edges (indexed by y-intervals)
  - `False`: Build tree for horizontal edges (indexed by x-intervals)

**Returns:**
- `IntervalTree`: Each interval contains `(edge_index, coordinate)` data
  - For vertical tree: intervals are y-ranges, coordinate is x-value
  - For horizontal tree: intervals are x-ranges, coordinate is y-value

### `ray_shooting(point, tree, is_left, is_down, is_ray_horizontal)`
Performs ray shooting from a point to find the nearest intersection with edges.

**Parameters:**
- `point`: `(x, y)` coordinates of ray origin
- `tree`: IntervalTree from `create_interval_tree()`
- `is_left`: Boolean - shoot ray leftward (if horizontal) 
- `is_down`: Boolean - shoot ray downward (if vertical)
- `is_ray_horizontal`: Boolean - ray direction
  - `True`: Horizontal ray (left/right)
  - `False`: Vertical ray (up/down)

**Returns:**
- `(intersection_point, bisected_edge_index)`: Tuple containing:
  - `intersection_point`: `(x, y)` where ray hits edge
  - `bisected_edge_index`: Index of the intersected edge

**Ray directions:**
- Horizontal + `is_left=True`: Shoot left (negative x)
- Horizontal + `is_left=False`: Shoot right (positive x)  
- Vertical + `is_down=True`: Shoot down (negative y)
- Vertical + `is_down=False`: Shoot up (positive y)


# DCEL Module (`orthogonal_dcel.py`)

## Core Classes

### `Vertex`
Represents a 2D point with incident half-edges.

**Attributes:**
- `x, y`: Coordinates
- `incident_half_edges`: List of outgoing half-edges
- `coords`: Property returning `(x, y)` tuple

**Methods:**
- `sort_incident()`: Orders incident edges counter-clockwise by angle

### `HalfEdge`
Directed edge in the DCEL with full connectivity information.

**Attributes:**
- `origin, destination`: Vertex endpoints
- `twin`: Opposite half-edge
- `next, prev`: Adjacent edges around face boundary
- `face`: Face this edge bounds
- `angle`: Cached angle with positive x-axis

**Methods:**
- `_is_point_to_left(point)`: Tests if point lies left of directed edge

### `Face`
Represents a face (bounded region) in the planar subdivision.

**Attributes:**
- `start_half_edge`: Representative edge of outer boundary
- `inner_components_half_edges`: List of hole boundaries
- `is_external`: Boolean indicating external (unbounded) face
- `height`: Optional height attribute for 3D extrusion

**Methods:**
- `outer_vertices()`: Iterator over outer boundary vertices
- `inner_vertices()`: Iterator over hole vertices (list per hole)
- `outer_edges()`: Iterator over outer boundary edges
- `inner_edges()`: Iterator over hole edges (list per hole)
- `is_vertex_outer(vertex)`: Test if vertex is on outer boundary

### `DCEL`
Main class managing the complete planar subdivision.

**Attributes:**
- `vertices`: List of all vertices
- `half_edges`: List of all half-edges
- `faces`: List of all faces

## DCEL Construction Methods

### `compute_faces_from_graph(G)`
Builds complete DCEL from NetworkX graph with orthogonal edges.

**Parameters:**
- `G`: NetworkX graph where edges have `is_vertical` boolean attribute

**Process:**
1. Creates vertices and half-edges from graph
2. Sets next/prev pointers using counter-clockwise vertex ordering
3. Finds all cycles in the half-edge structure
4. Classifies cycles as inner (clockwise) or outer (counter-clockwise)
5. Uses ray shooting to determine face containment relationships
6. Constructs faces with proper hole relationships

**Returns:** Self (for method chaining)

### `face_with_point(point, edges, tree, is_ray_horizontal)`
Determines which face contains a given point using ray shooting.

**Parameters:**
- `point`: Query point coordinates
- `edges`: Edge list (vertical or horizontal)
- `tree`: Corresponding interval tree
- `is_ray_horizontal`: Ray direction for shooting

**Returns:** Face object containing the point

## Visualization and I/O

### `plot_histogram_polyhedron(alpha=0.15, show_wireframe=True, face_colors=None)`
Creates 3D visualization of histogram polyhedron from face heights.

**Parameters:**
- `alpha`: Face transparency (0-1)  
- `show_wireframe`: Show edge outlines
- `face_colors`: Dict mapping faces to colors (auto-colored if None)

**Features:**
- Generates bottom faces, top faces, and side walls
- Handles faces with holes (inner boundaries)
- Color-codes faces by height using viridis colormap
- Returns matplotlib figure and axis objects

### File I/O Methods
- `save_to_pickle(filename)`: Serialize DCEL to binary file
- `load_from_pickle(filename)`: Class method to deserialize DCEL
- `convert_to_graph()`: Export DCEL back to NetworkX format

## Internal Algorithms

The DCEL construction uses several internal algorithms:

### Computing Cycles
- **`_add_vertex(x,y)`**: Adds a point (x,y) as a vertex of the DCEL and returns this vertex 
- **`_add_edge_pair(v1,v2)`**: Adds two half edges (v1,v2) and (v2,v1) to the DCEL and returns these two half edges
- **`_set_next_prev_pointers`**: Sets the previous and next pointers by iterating through each vertex and sorting the edges incident on this vertex counterclockwise 
- **`_find_all_cycles`**: Computes cycles by walking along `next` pointers

### Cycle Classification
- **`_is_inner_cycle(cycle)`**: Determines if cycle is clockwise (inner) or counter-clockwise (outer)
- Uses cross product at leftmost-bottommost vertex (found using `_left_bottom_edge(cycle)`) to compute interior angle

### Face-Cycle Assignment  
- **`_compute_connectivity_graph(cycles, vertical_edges, vertical_edges_tree)`**: 
  - Shoots horizontal rays left from the leftmost-bottommost outer cycle vertices
    - Uses  `_left_bottom_edge(cycle)` to compute such vertices
    - Uses `_determine_cycle_with_edge(cycles, edge, point)` to compute the cycle containing the half edge corresponding to `edge` which has `point` to its left
  - Constructs a graph on the set of cycles with edges between them if the ray from one cycle intersects the other

- **`_construct_faces_from_cycles`**:
  - Constructs the cycle connectivity graph using `_compute_connectivity_graph(cycles, vertical_edges, vertical_edges_tree)`
  - Determines which inner cycle contains each outer cycle using connected components of this graph

### 3D Polygon Generation
- **`_generate_3d_polygons()`**: Creates all faces of 3D histogram polyhedron
  - Bottom and top faces from 2D face boundaries
  - Vertical side walls connecting bottom/top vertices  
  - Inner walls for faces with holes
  - Proper orientation for correct normals
- **`_get_max_height`**: returns the maximum height of a face of the DCEL
- **`_set_axis_properties_3d(ax)`**: sets the x, y, and z-limits for illustration of the input DCEL

# Histogram Polyhedra Generator (`generate_histogram_polyhedra.py`)

## Configuration Parameters

```python
# Bounding box dimensions
box_width = 100
box_length = 100  
box_height = 100

# Rectangle generation
max_number_of_rectangles = 10
min_width = 1
max_width = box_width/2      # Up to 50 units
min_length = 1  
max_length = box_length/2    # Up to 50 units
min_height = 1
max_height = box_height      # Up to 100 units

# Output
num_instances = 999
output_dir = "Histogram_Polyhedra_Instances"
```

## Algorithm Steps

### 1. Rectangle Generation
- Creates $r$ random rectangles where $1 \leq r \leq 10$ within the $100 \times 100$ bounding box
- Each rectangle has random width/length in $[1, 50]$ units
- Rectangles can overlap arbitrarily
- Always includes the full bounding box as the outer boundary

### 2. PSLG Construction
The `convert_to_PSLG()` function:
- Converts rectangle boundaries to graph edges
- Handles high-degree vertices by creating edge chains
- Computes all rectangle intersections (crossings and touches)
- Creates intersection vertices and splits edges accordingly

### 3. DCEL Creation
- Converts the PSLG into a DCEL data structure using `orthogonal_dcel.py`
- Identifies all faces (bounded regions) in the planar subdivision
- Maintains topological relationships between vertices, edges, and faces

### 4. Height Assignment
- Assigns random heights $h \in [1, 100]$ to each face
- External (unbounded) face gets height $h = 0$
- Creates 3D histogram polyhedra from 2D face decomposition

### 5. Pickling
- Saves the DCEL as a `.pkl` file in `Histogram_Polyhedra_Instances`

## Key Functions

### `generate_random_rectangle()`
Creates a random axis-aligned rectangle within the bounding box.

### `resolve_high_degree_vertices(G)`
Handles vertices with degree $d > 2$ by:
- Classifying neighbors by direction (top, bottom, left, right)
- Converting star configurations into chains
- Maintaining orthogonal edge structure

### `find_all_intersections(G)`
Processes all edge intersections:
- **Crossings**: Creates new vertex, splits both edges into 4 segments
- **T-junctions**: Splits one edge, maintains the other
- Updates graph structure incrementally

### `convert_to_PSLG(rectangles)`
Main conversion pipeline:
1. Add rectangle boundary edges to graph
2. Resolve high-degree vertices  
3. Compute and process all intersections
4. Return clean PSLG

## Usage Examples

### Basic DCEL Construction

```python
import networkx as nx
from orthogonal_dcel import DCEL

# Create orthogonal graph
G = nx.Graph()
G.add_edge((0,0), (0,10), is_vertical=True)
G.add_edge((0,10), (10,10), is_vertical=False)
G.add_edge((10,10), (10,0), is_vertical=True)
G.add_edge((10,0), (0,0), is_vertical=False)

# Build DCEL
dcel = DCEL()
dcel.compute_faces_from_graph(G)

print(f"DCEL has {len(dcel.vertices)} vertices, {len(dcel.half_edges)} half-edges, {len(dcel.faces)} faces")
```

### 3D Histogram Visualization

```python
# Load saved instance
dcel = DCEL.load_from_pickle('instance_001.pkl')

# Visualize 3D histogram
fig, ax = dcel.plot_histogram_polyhedron(alpha=0.15, show_wireframe=True)
plt.show()
```

### Point Location Query

```python
from orthogonal_ray_shooting import get_vertical_and_horizontal_edges, create_interval_tree

# Convert DCEL to graph and build trees
G = dcel.convert_to_graph()
vertical_edges, horizontal_edges = get_vertical_and_horizontal_edges(G)
vertical_tree = create_interval_tree(vertical_edges, is_vertical_tree=True)

# Find face containing point
query_point = (5.5, 7.2)
containing_face = dcel.face_with_point(query_point, vertical_edges, vertical_tree, is_ray_horizontal=True)
print(f"Point {query_point} is in face with height {getattr(containing_face, 'height', 'undefined')}")
```

## Output

### Generated Instance Files
Generates 999 pickled DCEL instances in `Histogram_Polyhedra_Instances/`:
- `instance_001.pkl` through `instance_999.pkl`
- Each file contains a complete DCEL with face heights
- Files can be loaded using `DCEL.load_from_pickle(filename)`

### Running the Generator

```bash
python generate_histogram_polyhedra.py
```

The script will:
1. Create the output directory if it doesn't exist
2. Generate 999 random instances
3. Print progress: instance number and DCEL summary
4. Save each instance as a binary pickle file

---

# Decomposition into Boxes (`decompose_histogram_polyhedra.py`)

## Configuration Parameters

```python
# Bounding box dimensions (must match generator)
box_width = 100
box_height = 100

# Decomposition options
want_optimal = True          # True for optimal, False for 2-approximation
is_vertical_decomposition = True  # True for vertical, False for horizontal
```

## Algorithm Steps

### 1. Instance Loading
- Reads a random (or specified) instance from `Histogram_Polyhedra_Instances/`
- Converts DCEL to NetworkX graph representation
- Preserves face height information in `half_edge_height_dict`

### 2. Reflex Vertex Classification
Identifies four types of reflex vertices based on degree-2 vertex orientations:
- **Left-reflex**: Has neighbor to the right
- **Right-reflex**: Has neighbor to the left  
- **Bottom-reflex**: Has neighbor above
- **Top-reflex**: Has neighbor below

A vertex $v$ is classified as reflex if it has degree 2 and creates a "reflex angle" in the polygon boundary.

### 3. Good Edge Computation (Optimal Algorithm)
When `want_optimal = True`:
- Shoots rays from reflex vertices to find "good edges" (chords between reflex vertices)
- Uses interval trees for efficient $O(\log n)$ ray-shooting queries
- Computes maximum independent set of non-intersecting good edges
- Adds selected edges to decompose the PSLG optimally

The maximum independent set problem on intersection graphs of axis-aligned segments can be solved optimally using bipartite matching.

### 4. Vertical/Horizontal Decomposition
Performs systematic decomposition based on `is_vertical_decomposition`:

**Vertical Decomposition:**
- Shoots rays downward from bottom-reflex vertices
- Shoots rays upward from top-reflex vertices

**Horizontal Decomposition:**
- Shoots rays leftward from left-reflex vertices  
- Shoots rays rightward from right-reflex vertices

### 5. Graph Updates
- Adds intersection points as new vertices
- Splits bisected edges appropriately
- Maintains height information for new edges
- Updates DCEL structure with decomposed geometry

## Key Functions

### `is_*_reflex(v)` Functions
```python
def is_left_reflex(v):
    # Returns True if vertex v is left-reflex
    if G.degree(v) == 2:
        u, w = G.neighbors(v)
        return v[0] < u[0] or v[0] < w[0]
```

### `compute_good_edges()`
Finds all potential good edges by:
- Ray shooting from each reflex vertex in appropriate direction
- Checking if ray terminates at another vertex (not edge interior)
- Returns list of good edges with orientation information

### `find_max_IS_of_good_edges()`
Computes maximum independent set of good edges:
- Models intersection conflicts as bipartite graph $H = (V_h \cup V_v, E)$
- Uses maximum matching to find minimum vertex cover of size $|M|$
- Returns complement as maximum independent set of size $|V| - |M|$

### `ray_shooting_from_a_set_of_reflex_vertices()`
Performs batch ray shooting:
- Shoots rays from specified reflex vertex set
- Uses interval trees for efficient $O(\log n)$ intersection queries
- Sorts results by bisected edge and intersection coordinate

### `add_ray_points_to_graph()`
Updates graph structure with ray intersections:
- Adds intersection points as new vertices
- Splits bisected edges into segments
- Maintains height information consistency
- Removes original edges after subdivision

## Visualization Features

### Input Visualization
- Plots original PSLG with reflex vertices color-coded:
  - **Red**: Left-bottom reflex
  - **Orange**: Right-bottom reflex  
  - **Green**: Left-top reflex
  - **Brown**: Right-top reflex
- Shows 3D histogram polyhedron

### Decomposition Visualization
- Displays decomposed PSLG with added chords
- Highlights good edges in purple (optimal algorithm)
- Maintains reflex vertex color coding
- Shows final 3D decomposed histogram

## Usage Examples

### Basic Decomposition (2-Approximation)

```python
# Set parameters
want_optimal = False
is_vertical_decomposition = True

# Load and decompose
python decompose_histogram_polyhedra.py
```

### Optimal Decomposition

```python
# Set parameters
want_optimal = True
is_vertical_decomposition = True  # or False for horizontal

# Run decomposition
python decompose_histogram_polyhedra.py
```

### Specific Instance Selection

```python
# Choose specific instance instead of random
chosen_file = "instance_620.pkl"  # Nice instances: 985, 074, 672, 253, 146, 444, 839, 896, 856
```

## Output Analysis

The script provides detailed output including:

### Console Output
```
instance_XXX.pkl

Input DCEL
Face_0 height: 0
Face_1 height: 45
...

Maximum Independent Set of good edges: [((x1,y1), (x2,y2)), ...]

Final DCEL  
Face_0 height: 0
Face_1 height: 45
...
```

### Visual Output
1. **Original PSLG**: Shows input histogram with reflex vertices
2. **3D Original**: Displays input histogram polyhedron
3. **Decomposed PSLG**: Shows PSLG after adding decomposition chords
4. **3D Decomposed**: Displays final decomposed histogram polyhedron

## Algorithm Complexity

### Optimal Algorithm
- **Good edge computation**: $O(n \log n)$ per reflex vertex
- **Maximum independent set**: $O(k^3)$ where $k$ is number of good edges
- **Overall**: $O(n \log n + k^3)$ where $n$ is number of vertices

### 2-Approximation Algorithm  
- **Ray shooting**: $O(n \log n)$ per reflex vertex type
- **Graph updates**: $O(n)$ per new intersection
- **Overall**: $O(n \log n)$ 

## Running the Decomposition

```bash
python decompose_histogram_polyhedra.py
```

The script will:
1. Load a random instance from `Histogram_Polyhedra_Instances/`
2. Display the chosen filename
3. Show input DCEL face information
4. Perform decomposition based on configuration
5. Display decomposition results
6. Generate multiple visualization plots
7. Show final decomposed DCEL information
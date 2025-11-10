import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as plt
import random
from format_logger import setup_logger
import logging
from skimage import measure
import numpy as np

# Setup logger
setup_logger(level="INFO")
logger = logging.getLogger(__name__)

# Parameters
shape = (25, 25, 25)
density = random.uniform(0.1, 0.4)  # Density of the polyhedra

class Vertex:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class Edge:
    def __init__(self, v1: Vertex, v2: Vertex):
        self.v1 = v1
        self.v2 = v2

class Face:
    def __init__(self, vertices: list[Vertex]):
        self.vertices = vertices
        self.edges = [Edge(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]
        self.normal = self.compute_normal()
    def compute_normal(self):
        if len(self.vertices) < 3:
            return None
        v1 = np.array([self.vertices[1].x - self.vertices[0].x,
                       self.vertices[1].y - self.vertices[0].y,
                       self.vertices[1].z - self.vertices[0].z])
        v2 = np.array([self.vertices[2].x - self.vertices[1].x,
                       self.vertices[2].y - self.vertices[1].y,
                       self.vertices[2].z - self.vertices[1].z])
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm == 0:
            return normal
        return normal / norm

class OrthogonalPolyhedra:
    def __init__(self, vertices: list[Vertex], faces: list[Face]):
        self.vertices = vertices
        self.faces = faces
        self.edges = self.extract_edges()
    
    def extract_edges(self) -> list[Edge]:
        edge_set = set()
        for face in self.faces:
            for edge in face.edges:
                edge_tuple = tuple(sorted([(edge.v1.x, edge.v1.y, edge.v1.z), (edge.v2.x, edge.v2.y, edge.v2.z)]))
                edge_set.add(edge_tuple)
        return [Edge(Vertex(*e[0]), Vertex(*e[1])) for e in edge_set]

def create_simple_voxel_polyhedra(shape: tuple, density: float) -> np.ndarray:
    """Create a simple connected 3D polyhedra using voxels."""
    # Generate random 3D matrix
    matrix = (np.random.rand(*shape) < density)

    # Ensure connectivity: keep only the largest connected component
    structure = np.array([
        [[0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]],
        
        [[0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]],
        
        [[0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]]
    ], dtype=bool)

    labeled, num_features = label(matrix, structure=structure)
    if num_features > 0:
        # Find the largest component
        largest_label = np.argmax(np.bincount(labeled.flat)[1:]) + 1
        matrix = (labeled == largest_label)
    else:
        matrix = np.zeros(shape, dtype=bool)
    
    return matrix

def move_to_corner(matrix: np.ndarray) -> np.ndarray:
    """Move the True voxels to the bottom-left corner (0,0,0)"""
    coords = np.argwhere(matrix)
    
    if len(coords) == 0:
        return matrix
    
    # Find bounding box
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    
    # Extract the bounding box
    slices = tuple(slice(min_c, max_c + 1) for min_c, max_c in zip(min_coords, max_coords))
    cropped = matrix[slices]
    
    # Create new matrix and place cropped region at origin
    new_matrix = np.zeros_like(matrix)
    new_slices = tuple(slice(0, max_c - min_c + 1) for min_c, max_c in zip(min_coords, max_coords))
    new_matrix[new_slices] = cropped
    
    return new_matrix

import numpy as np

def convert_voxels_to_mesh(matrix):
    """
    Extract vertices, edges, and faces for the surface of a voxel polyhedron.
    Returns faces as quads (4 vertices per face) aligned with voxel boundaries.
    """
    vertices = []
    faces = []
    vertex_dict = {}  # Map (x,y,z) -> vertex index
    
    def get_or_create_vertex(pos):
        """Get vertex index, creating it if it doesn't exist"""
        pos_tuple = tuple(pos)
        if pos_tuple not in vertex_dict:
            vertex_dict[pos_tuple] = len(vertices)
            vertices.append(pos)
        return vertex_dict[pos_tuple]
    
    # For each True voxel
    coords = np.argwhere(matrix)
    
    for x, y, z in coords:
        # Check all 6 faces of this voxel
        # Face is exposed if neighbor in that direction is False/outside bounds
        
        # -X face (left)
        if x == 0 or not matrix[x-1, y, z]:
            v0 = get_or_create_vertex([x, y, z])
            v1 = get_or_create_vertex([x, y+1, z])
            v2 = get_or_create_vertex([x, y+1, z+1])
            v3 = get_or_create_vertex([x, y, z+1])
            faces.append([v0, v1, v2, v3])
        
        # +X face (right)
        if x == matrix.shape[0]-1 or not matrix[x+1, y, z]:
            v0 = get_or_create_vertex([x+1, y, z])
            v1 = get_or_create_vertex([x+1, y, z+1])
            v2 = get_or_create_vertex([x+1, y+1, z+1])
            v3 = get_or_create_vertex([x+1, y+1, z])
            faces.append([v0, v1, v2, v3])
        
        # -Y face (front)
        if y == 0 or not matrix[x, y-1, z]:
            v0 = get_or_create_vertex([x, y, z])
            v1 = get_or_create_vertex([x, y, z+1])
            v2 = get_or_create_vertex([x+1, y, z+1])
            v3 = get_or_create_vertex([x+1, y, z])
            faces.append([v0, v1, v2, v3])
        
        # +Y face (back)
        if y == matrix.shape[1]-1 or not matrix[x, y+1, z]:
            v0 = get_or_create_vertex([x, y+1, z])
            v1 = get_or_create_vertex([x+1, y+1, z])
            v2 = get_or_create_vertex([x+1, y+1, z+1])
            v3 = get_or_create_vertex([x, y+1, z+1])
            faces.append([v0, v1, v2, v3])
        
        # -Z face (bottom)
        if z == 0 or not matrix[x, y, z-1]:
            v0 = get_or_create_vertex([x, y, z])
            v1 = get_or_create_vertex([x+1, y, z])
            v2 = get_or_create_vertex([x+1, y+1, z])
            v3 = get_or_create_vertex([x, y+1, z])
            faces.append([v0, v1, v2, v3])
        
        # +Z face (top)
        if z == matrix.shape[2]-1 or not matrix[x, y, z+1]:
            v0 = get_or_create_vertex([x, y, z+1])
            v1 = get_or_create_vertex([x, y+1, z+1])
            v2 = get_or_create_vertex([x+1, y+1, z+1])
            v3 = get_or_create_vertex([x+1, y, z+1])
            faces.append([v0, v1, v2, v3])

    vertices = np.array(vertices)
    faces = np.array(faces)
    
    logger.info(f"Extracted {len(vertices)} vertices and {len(faces)} faces from voxel matrix.")
    logger.info(f"\nVertices:")
    for i, v in enumerate(vertices):
        logger.info(f"  {i}: {v}")
    logger.info(f"\nFaces:")   
    for i, f in enumerate(faces):
        logger.info(f"  {i}: {f}")

    return vertices, faces

def clean_mesh(vertices, faces):
    x_sorted_vertices = vertices[np.lexsort((vertices[:,2], vertices[:,1], vertices[:,0]))]
def plot_voxel_polyhedra(matrix: np.ndarray) -> None:
    """Plot the voxel polyhedra."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(matrix, facecolors='red', edgecolor='k')

    ax.set_box_aspect([1,1,1])  # Equal aspect ratio

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Random Connected 3D Polyhedra from Voxels')

def plot_mesh_polyhedra(verts, faces) -> None:
    """Plot the mesh polyhedra."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a 3D polygon collection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    mesh = Poly3DCollection(verts[faces], alpha=0.7)
    face_color = (0.5, 0.5, 1)
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    # Auto scale to the mesh size
    scale = verts.flatten()
    ax.auto_scale_xyz(scale, scale, scale)

    ax.set_box_aspect([1,1,1])  # Equal aspect ratio

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Mesh Polyhedra from Voxels')

if __name__ == "__main__":
    logger.info("Generating random connected 3D polyhedra...")
    matrix = create_simple_voxel_polyhedra(shape, density)
    logger.info("Moving polyhedra to corner...")
    matrix = move_to_corner(matrix)
    logger.info("Plotting the voxel polyhedra...")
    plot_voxel_polyhedra(matrix)
    logger.info("Converting voxels to mesh...")
    verts, faces = convert_voxels_to_mesh(matrix)
    logger.info("Plotting the mesh polyhedra...")
    plot_mesh_polyhedra(verts, faces)
    plt.show()
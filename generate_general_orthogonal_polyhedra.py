import numpy as np
import random
from solid2 import *
from solid2 import scad_render_to_file
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import networkx as nx

def generate_random_box(max_x=100, max_y=100, max_z=100):
    """Generate a random box with integer coordinates within the environment"""
    # Random position (ensure box fits within environment)
    max_size = min(max_x, max_y, max_z) // 3  # Maximum box dimension (1/5 of smallest env dimension)
    min_size = 1          # Minimum box dimension
    
    # Generate box dimensions
    width = random.randint(min_size, min(max_size, max_x))
    height = random.randint(min_size, min(max_size, max_y))
    depth = random.randint(min_size, min(max_size, max_z))
    
    # Generate position (ensure box stays within bounds)
    x = random.randint(0, max_x - width)
    y = random.randint(0, max_y - height)
    z = random.randint(0, max_z - depth)
    
    return {
        'position': (x, y, z),
        'dimensions': (width, height, depth),
        'bounds': ((x, x + width), (y, y + height), (z, z + depth))
    }

def boxes_overlap(box1, box2):
    """Check if two boxes overlap"""
    b1 = box1['bounds']
    b2 = box2['bounds']
    
    # Check overlap in all three dimensions
    x_overlap = b1[0][0] < b2[0][1] and b2[0][0] < b1[0][1]
    y_overlap = b1[1][0] < b2[1][1] and b2[1][0] < b1[1][1]
    z_overlap = b1[2][0] < b2[2][1] and b2[2][0] < b1[2][1]
    
    return x_overlap and y_overlap and z_overlap

def find_connected_components(boxes):
    """Find groups of overlapping boxes using NetworkX graph"""
    # Create a graph with boxes as nodes
    G = nx.Graph()
    G.add_nodes_from(range(len(boxes)))
    
    # Add edges between overlapping boxes
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes_overlap(boxes[i], boxes[j]):
                G.add_edge(i, j)
    
    # Find connected components
    components = []
    for component_indices in nx.connected_components(G):
        component_boxes = [boxes[i] for i in component_indices]
        components.append(component_boxes)
    
    return components

def create_openscad_box(box):
    """Create an OpenSCAD box object"""
    pos = box['position']
    dims = box['dimensions']
    return translate(pos)(cube(dims))

def create_union_of_boxes(boxes):
    """Create a union of multiple boxes"""
    if len(boxes) == 1:
        return create_openscad_box(boxes[0])
    
    result = create_openscad_box(boxes[0])
    for box in boxes[1:]:
        result = result + create_openscad_box(box)
    
    return result

def generate_polyhedron(num_boxes=20, max_x=100, max_y=100, max_z=100):
    """Generate the complete polyhedron"""
    print(f"Generating {num_boxes} random boxes in {max_x}x{max_y}x{max_z} environment...")
    
    # Generate random boxes
    boxes = [generate_random_box(max_x, max_y, max_z) for _ in range(num_boxes)]
    
    # Find connected components
    print("Finding connected components...")
    components = find_connected_components(boxes)
    print(f"Found {len(components)} connected components")
    
    # Create the main environment box
    env_box = cube([max_x, max_y, max_z])
    
    # Create unions of each connected component
    unions = []
    for i, component in enumerate(components):
        print(f"Component {i+1}: {len(component)} boxes")
        union_obj = create_union_of_boxes(component)
        unions.append(union_obj)
    
    # Subtract all unions from the environment box
    result = unions[0]
    for union_obj in unions:
        result = result + union_obj
    
    return result, boxes, components

def save_and_render_polyhedron(max_x=100, max_y=100, max_z=100, num_boxes=15):
    """Generate and save the polyhedron"""
    # Generate the polyhedron
    polyhedron, boxes, components = generate_polyhedron(num_boxes, max_x, max_y, max_z)
    
    # Save to OpenSCAD file
    scad_render_to_file(polyhedron, 'polyhedron.scad')
    print("OpenSCAD file saved as 'polyhedron.scad'")
    
    # Print statistics
    print(f"\nPolyhedron Statistics:")
    print(f"Environment size: {max_x}x{max_y}x{max_z}")
    print(f"Total boxes generated: {len(boxes)}")
    print(f"Connected components: {len(components)}")
    print(f"Component sizes: {[len(comp) for comp in components]}")
    
    return boxes, components

def visualize_boxes(boxes, components, max_x=100, max_y=100, max_z=100):
    """Visualize the box arrangement before subtraction"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color palette for different components
    colors = plt.cm.Set3(np.linspace(0, 1, len(components)))
    
    # Draw environment box wireframe
    env_vertices = np.array([
        [0, 0, 0], [max_x, 0, 0], [max_x, max_y, 0], [0, max_y, 0],
        [0, 0, max_z], [max_x, 0, max_z], [max_x, max_y, max_z], [0, max_y, max_z]
    ])
    
    # Environment box edges
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # top
        [0, 4], [1, 5], [2, 6], [3, 7]   # sides
    ]
    
    for edge in edges:
        points = env_vertices[edge]
        ax.plot3D(*points.T, 'k--', alpha=0.3, linewidth=0.5)
    
    # Draw each component with a different color
    for comp_idx, component in enumerate(components):
        color = colors[comp_idx]
        
        for box in component:
            pos = box['position']
            dims = box['dimensions']
            
            # Create box vertices
            x, y, z = pos
            w, h, d = dims
            
            vertices = np.array([
                [x, y, z], [x+w, y, z], [x+w, y+h, z], [x, y+h, z],
                [x, y, z+d], [x+w, y, z+d], [x+w, y+h, z+d], [x, y+h, z+d]
            ])
            
            # Define the 6 faces of the box
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
                [vertices[4], vertices[7], vertices[6], vertices[5]],  # top
                [vertices[0], vertices[4], vertices[5], vertices[1]],  # front
                [vertices[2], vertices[6], vertices[7], vertices[3]],  # back
                [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
                [vertices[1], vertices[5], vertices[6], vertices[2]]   # right
            ]
            
            # Add faces to plot
            poly3d = [[tuple(vertex) for vertex in face] for face in faces]
            ax.add_collection3d(Poly3DCollection(poly3d, facecolor=color, alpha=0.7, edgecolor='black', linewidth=0.5))
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.set_zlim(0, max_z)
    ax.set_title(f'Random Boxes Before Subtraction ({max_x}x{max_y}x{max_z})\n(Different colors = different connected components)')
    
    plt.tight_layout()
    plt.show()
    
    return fig

if __name__ == "__main__":
    # Environment dimensions (easily customizable)
    ENV_X, ENV_Y, ENV_Z = 100, 100, 100
    NUM_BOXES = 20
    
    # Generate and save the polyhedron
    boxes, components = save_and_render_polyhedron(ENV_X, ENV_Y, ENV_Z, NUM_BOXES)
    
    # Visualize the box arrangement
    print("\nGenerating visualization...")
    visualize_boxes(boxes, components, ENV_X, ENV_Y, ENV_Z)
    
    print(f"\nTo view the final polyhedron:")
    print("1. Open 'polyhedron.scad' in OpenSCAD")
    print("2. Press F5 to preview or F6 to render")
    print(f"3. The result will be a {ENV_X}x{ENV_Y}x{ENV_Z} box with cavities where the random boxes were subtracted")
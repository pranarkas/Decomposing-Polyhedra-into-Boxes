import random
from shapely.geometry import box
import json
import os

# Dimensions of the large bounding box
box_width = 100
box_height = 100

# Number of rectangles to generate
max_number_of_rectangles = 10

# Size constraints for small rectangles
min_width = 1
max_width = box_width/2
min_height = 1
max_height = box_height/2

def generate_random_rectangle():
    width = random.randint(min_width, max_width)
    height = random.randint(min_height, max_height)
    x = random.randint(0, box_width - width)
    y = random.randint(0, box_height - height)
    return box(x, y, x + width, y + height)

def write_rectangles_to_file(rectangles, filename):
    all_coords = [list(r.exterior.coords) for r in rectangles]
    with open(filename, "w") as f:
        json.dump(all_coords, f)

output_dir = "Instances"
num_instances = 999

for i in range(1, num_instances + 1):
    num_rectangles = random.randint(1,max_number_of_rectangles)
    rectangles = []
    while len(rectangles) < num_rectangles:
        new_rectangle = generate_random_rectangle()
        rectangles.append(new_rectangle)
    
    filename = os.path.join(output_dir, f"instance_{i:03}.json")
    write_rectangles_to_file(rectangles, filename)
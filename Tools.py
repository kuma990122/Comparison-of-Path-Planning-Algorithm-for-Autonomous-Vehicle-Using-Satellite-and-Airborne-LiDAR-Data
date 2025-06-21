import math
import numpy as np
import time
import heapq
import random
import tracemalloc

# from 'exists' to 'heuristics' are 2D Path Planning Methods's general tool functions
# checks if node exists
def exists(img, x, y, width, height):
    if x < width and x >= 0 and y < height and y >= 0:
        return True
    else:
        return False

# checks if the node is traversable
def is_traversable(img, x, y):
    r = img[y][x][0]
    g = img[y][x][1]
    b = img[y][x][2]
    # white and green path
    if r >= 250 and g >= 250 and b >= 250:
        return True
    if r < 10 and g > 220 and b < 10:
        return True
    else:
        return False

def ifDiagonal(cur_x, cur_y, n_x, n_y):
    neighbor = [n_x,n_y]
    current = (cur_x, cur_y)
    if(   neighbor ==[current[0]-1, current[1]-1]
       or neighbor ==[current[0]-1, current[1]+1]
       or neighbor ==[current[0]+1, current[1]+1]
       or neighbor ==[current[0]+1, current[1]-1]):
        return True
    
    else:
        return False
# checks if the destination has been reached
def is_destination(img, x, y, ending_position):
    if x == ending_position[0] and y == ending_position[1]:
        return True
    else:
        return False

def E_distance(node1_x, node1_y, node2_x, node2_y):

    e_dis = math.sqrt((node2_x -node1_x)**2 + (node2_y - node1_y)**2)

    return e_dis
# calculates heuristics (diagonal)
def heuristics(img, x, y, ending_position):
     # getting V/H distance
    r = img[y][x][0]
    g = img[y][x][1]
    b = img[y][x][2]
    if r >= 250 and g >= 250 and b >= 250:
        d1 = 100  # 1 * 10
        x_end = ending_position[0]
        y_end = ending_position[1]
        x_distance = abs(x - x_end)
        y_distance = abs(y - y_end)

        # getting diagonal distance
        d2 = 140  # sqrt(2) * 10
        diag_distance = abs(x_distance - y_distance)
        #diag_distance = math.sqrt(x_distance * x_distance + y_distance * y_distance)
        h = min(x_distance, y_distance) * d1 + diag_distance * d2

        #diag_distance = math.sqrt(x_distance * x_distance + y_distance * y_distance)
        #h = (x_distance + y_distance) * d1 + diag_distance * d2

        #h = (x_distance + y_distance) * d1 + min(x_distance, y_distance) * (d2 - 2 * d1)

    elif r < 10 and g > 220 and b < 10:
        d1 =1  # 1 * 10
        x_end = ending_position[0]
        y_end = ending_position[1]
        x_distance = abs(x - x_end)
        y_distance = abs(y - y_end)

        # getting diagonal distance
        d2 = 1.4 # sqrt(2) * 10
        #diag_distance = math.sqrt(x_distance * x_distance + y_distance *y_distance)
        diag_distance = abs(x_distance - y_distance)
        h = min(x_distance, y_distance) * d1 + diag_distance * d2

        #h = (x_distance+ y_distance) * d1 + diag_distance * d2

        #h = (x_distance + y_distance) * d1 +min(x_distance, y_distance) *(d2 - 2 * d1)

    return h

def heuristics3D(img,img_gray, x, y, ending_position, sample_size=3):
    """
    Calculate heuristic with sampling-based gradient estimation for path planning.
    """
    x_end, y_end = ending_position
    # basic heuristic calculation
    z_current = np.float64(img_gray[y, x]) 
    d1, d2 = (100, 140) if z_current >= 80 else (1, 1.4)

    # Use Diagonal Distance and path weight as the initial heuristic
    x_dist = abs(x - x_end)
    y_dist = abs(y - y_end)
    h_initial = min(x_dist, y_dist) * d1 + abs(x_dist - y_dist) * d2

    # local sampling, according to the value of sample_size 
    sample_points_list = [(x + i, y + j) for i in range(-sample_size, sample_size + 1)
                     for j in range(-sample_size, sample_size + 1)
                     if (i, j) != (0, 0)]
    
    sample_points = []
    for sample_point in sample_points_list:
        if is_traversable(img, sample_point[0], sample_point[1]): sample_points.append(sample_point) 
    
    # Calculate the future gradient of the path
    gradients = []
    for nx, ny in sample_points:
        if 0 <= nx < img_gray.shape[1] and 0 <= ny < img_gray.shape[0]:
            z_next = np.float64(img_gray[ny, nx]) 
            gradient = abs(np.clip(z_next - z_current, -1e10, 1e10)) / (np.sqrt((nx - x) ** 2 + (ny - y) ** 2) + 1e-5)
            gradients.append(gradient)
    
    avg_gradient = np.mean(gradients) if gradients else 0
    h = 0.8*h_initial + avg_gradient * 500  # increase the weight factor to make it works
    
    return h

def get_neighbors(img, x, y):
    all_neighbors = [(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx != 0 or dy != 0)]
    available_list = []
    for neighbor in all_neighbors:
        if exists(img, neighbor[0], neighbor[1], 1024, 1024) and is_traversable(img, neighbor[0], neighbor[1]):
            available_list.append(neighbor)
    return available_list

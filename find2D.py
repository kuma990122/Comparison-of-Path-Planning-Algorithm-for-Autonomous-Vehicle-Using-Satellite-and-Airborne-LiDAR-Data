import math
import numpy as np
import time
import heapq
import random
import tracemalloc
from Tools import exists, is_traversable, ifDiagonal, is_destination, E_distance, heuristics

t1 = time.perf_counter()

# Path planning function implementations
# A* alg
import heapq
import time
import tracemalloc
import numpy as np

def findbyAStar(img, width, height, starting_position, ending_position, path_color):
    tracemalloc.start()
    start_time = time.time()

    if is_destination(img, starting_position[0], starting_position[1], ending_position):
        exit()

    # Create an array to store the f, g, h values ​​of each node and the parent node coordinates
    node_details = np.full((width, height, 5), -1)

    # Create a 2D array marking the nodes that have been evaluated
    closed_list = np.zeros((width, height))

    # Create a priority queue (min heap) to store the nodes to be evaluated
    # Each element contains: [f, x, y]
    open_list = []
    heapq.heappush(open_list, (0, starting_position[0], starting_position[1]))

    while open_list:
        # Pop the node with the lowest f value
        current = heapq.heappop(open_list)
        cur_f, cur_x, cur_y = current

        r = img[cur_y][cur_x][0]
        g = img[cur_y][cur_x][1]
        b = img[cur_y][cur_x][2]

        # Mark the current node as evaluated
        closed_list[cur_x][cur_y] = 1
        
        if is_destination(img, cur_x, cur_y, ending_position):
            print("Goal is successfully found")
            img_Astar, cost_Astar = print_path(img, starting_position, ending_position, path_color, node_details, dim=2)
            end_time = time.time()
            current_mem, peak_mem = tracemalloc.get_traced_memory()
            print(f"Algorithm execute time: {end_time - start_time} s")
            print(f"peak memory usage: {peak_mem / 1024 / 1024:.2f} MB")
            tracemalloc.stop()
            return img_Astar, end_time - start_time, peak_mem, cost_Astar

        neighbor_list = [
            (cur_x - 1, cur_y), (cur_x + 1, cur_y), (cur_x, cur_y + 1), (cur_x, cur_y - 1),
            (cur_x - 1, cur_y + 1), (cur_x - 1, cur_y - 1), (cur_x + 1, cur_y + 1), (cur_x + 1, cur_y - 1)
        ]

        for direction_counter, (n_x, n_y) in enumerate(neighbor_list):
            if exists(img, n_x, n_y, width, height) and is_traversable(img, n_x, n_y) and closed_list[n_x][n_y] == 0:
                # calculate g value
                if direction_counter <= 3 and r >= 250 and g >= 250 and b >= 250:
                    g_cost = 100
                elif direction_counter > 3 and r >= 250 and g >= 250 and b >= 250:
                    g_cost = 140
                elif direction_counter <= 3 and r < 10 and g > 220 and b < 10:
                    g_cost = 1
                else:
                    g_cost = 1.4

                if (cur_x, cur_y) != starting_position:
                    g_cost += node_details[cur_x][cur_y][1]

                # calculate h and f value
                h = heuristics(img, n_x, n_y, ending_position)
                f = g_cost + h

                if node_details[n_x][n_y][0] == -1 or f < node_details[n_x][n_y][0]:
                    node_details[n_x][n_y] = [f, g_cost, h, cur_x, cur_y]
                    heapq.heappush(open_list, (f, n_x, n_y))

    exit()

# Dijkstra alg
def findbyDijkstra(img, width, height, starting_position, ending_position, path_color):
    tracemalloc.start()
    start_time = time.time()

    # Initialize distance table and predecessor dictionary
    distance = np.full((height, width), np.inf)
    distance[starting_position[1], starting_position[0]] = 0

    # Priority queue for unvisited nodes
    pq = []
    heapq.heappush(pq, (0, starting_position))  # (distance, (x, y))

    # Predecessor dictionary to reconstruct the path
    predecessors = {}

    while pq:
        cur_distance, current = heapq.heappop(pq)
        cur_x, cur_y = current

        # If we reached the ending position, break the loop
        if current == (ending_position[0], ending_position[1]):
            print("Path was successfully found.")
            break

        # Explore neighbors
        neighbor_list = [
            (cur_x - 1, cur_y), (cur_x + 1, cur_y), 
            (cur_x, cur_y + 1), (cur_x, cur_y - 1),
            (cur_x - 1, cur_y + 1), (cur_x - 1, cur_y - 1),
            (cur_x + 1, cur_y + 1), (cur_x + 1, cur_y - 1)
        ]
        
        for neighbor in neighbor_list:
            n_x, n_y = neighbor
            
            # Check if the neighbor is within bounds and traversable
            if exists(img, n_x, n_y, width, height) and is_traversable(img, n_x, n_y):
                r, g, b = img[n_y][n_x]

                # Calculate distance cost based on terrain (color)
                if ifDiagonal(cur_x, cur_y, n_x, n_y):
                    n_d = 1.4 if r < 10 and g > 220 and b < 10 else 140
                else:
                    n_d = 1 if r < 10 and g > 220 and b < 10 else 100
                
                # Calculate tentative distance to this neighbor
                tentative_distance = cur_distance + n_d

                # Only consider this neighbor if it offers a shorter path
                if tentative_distance < distance[n_y, n_x]:
                    distance[n_y, n_x] = tentative_distance
                    predecessors[(n_x, n_y)] = (cur_x, cur_y)  # Record predecessor
                    heapq.heappush(pq, (tentative_distance, (n_x, n_y)))
                    # print(tentative_distance)

    # Reconstruct the shortest path
    path = []
    current = (ending_position[0], ending_position[1])  # Use tuple for current
    while current in predecessors:
        path.append(current)
        current = predecessors[current]
    path.append((starting_position[0], starting_position[1]))  # Add starting position
    path.reverse()  # Reverse to get from start to end
    img_dij, cost_Dijkstra = print_pathDijk(img,starting_position,ending_position,path_color,predecessors)
    end_time = time.time()
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    print(f"Algorithm execution time: {end_time - start_time:.6f} sec")
    print(f"Peak memory usage: {peak_memory / 1024 / 1024:.2f} MB")
    tracemalloc.stop()
    return img_dij, end_time - start_time, peak_memory, cost_Dijkstra
# RRT* alg
def findbyrrtstar(img, width, height ,starting_position, ending_position, path_color):
    tracemalloc.start()
    start_time = time.time()
    
    # each node has [cost, parent_x, parent_y, x, y
    Tree = []
    # Step stride towards random node
    step_size = 3
    # radius range for new node
    radius = 5
    # Storing the path for latter usage
    path_list = []
    pathfind = False
    Tree.append([0,starting_position[0],starting_position[1],starting_position[0],starting_position[1]])
    while(pathfind != True):
        node_random = [random.randint(0, width - 1),random.randint(0, height - 1)]
        r, g, b = img[node_random[1]][node_random[0]]
        node_new = []
        # Choose the closest node to random node as the near node
        if (r >= 250 and g >= 250 and b >= 250) or (r < 10 and g > 220 and b < 10):
            closest_dis = np.inf
            closest_node = []
            for node in Tree:
                nodes_dis = E_distance(node[3],node[4], node_random[0], node_random[1])
                if(nodes_dis < closest_dis): 
                    closest_dis = nodes_dis
                    closest_node = node
            # expand the near node with the specified step size as new node
            if closest_dis != 0:
                # expand from closest node towards the random node with step size
                node_new_x = math.ceil(closest_node[3] + step_size * ((node_random[0] - closest_node[3]) / closest_dis))
                node_new_y = math.ceil(closest_node[4] + step_size * ((node_random[1] - closest_node[4]) / closest_dis))
                node_new = [node_new_x,node_new_y]
            else:
                continue
            if 0 <= node_new[0] < width and 0 <= node_new[1] < height:    
                if(is_traversable(img, node_new[0], node_new[1])):
                    r_new, g_new, b_new = img[node_new[1]][node_new[0]]

                    # Choose the parent node for new node, if the new node is an available node
                    if (r_new >= 250 and g_new >= 250 and b_new >= 250) or (r_new < 10 and g_new > 220 and b_new < 10):
                        new_cost = np.inf
                        new_parent = []
                        near_list = []
                        for node in Tree:
                            if(E_distance(node[3],node[4], node_new[0], node_new[1]) <= radius): 
                                near_list.append(node)                
                                cost2new = E_distance(node[3], node[4], node_new[0], node_new[1])
                                if(node[0]+cost2new < new_cost):
                                    new_cost = node[0] + cost2new
                                    new_parent = [node[3],node[4]]
                        new_node_details = [new_cost, new_parent[0], new_parent[1], node_new[0], node_new[1]]
                        Tree.append(new_node_details)
                        #print(f"new node cost: {new_node_details[0]}")
                        # Rewire the neighboring nodes
                        for node in near_list:
                            if node[3] == new_node_details[1] and node[4] == new_node_details[2]:
                                continue
                            cost_node2New = E_distance(node[3], node[4], new_node_details[3],new_node_details[4])
                            cost_conNew2Init = new_cost + cost_node2New
                            if(cost_conNew2Init < node[0]):
                                node[0] = cost_conNew2Init
                                node[1] = new_node_details[1]
                                node[2] = new_node_details[2]
                        if(E_distance(node_new[0], node_new[1], ending_position[0],ending_position[1]) <= step_size):
                            pathfind = True
                            print(f"The final cost: {new_cost}")
                            break
    # Trace the path from goal to start
    node = Tree[-1]
    while node[3] != starting_position[0] or node[4] != starting_position[1]:
        path_list.append([node[3], node[4]])  # Add current node to path
        parent_x, parent_y = node[1], node[2]

        # Find the parent node in the tree
        for n in Tree:
            if n[3] == parent_x and n[4] == parent_y:
                node = n
                break

    # Add starting and ending positions to the path
    path_list.append(starting_position)
    path_list.reverse()  # Reverse to get path from start to goal
    img_rrt, cost_RRTStar = printRRTstar(img, path_list, path_color)
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    print(f"RRT* Algorithm execution time: {end_time - start_time} sec")
    print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")
    tracemalloc.stop()
    return img_rrt, end_time - start_time, peak, cost_RRTStar

# Improved Ants Colony Optimization alg
def findbyNIACO(img, width, height, starting_position, ending_position, path_color):
    tracemalloc.start()
    start_time = time.time()
    pheromone_matrix = np.full((height, width), 1)
    pheromone_matrix = pheromone_matrix.astype(np.float64)
    ant_num = 10
    max_iteration = 5
    q = random.random()
    q0 = 1.0
    q0_weight = 0.4
    transfer_counter = 1
    starting_position = tuple(starting_position)
    ending_position = tuple(ending_position)
    #tabu_list = []
    path_list = []
    global_optimal = []
    global_optimal_length = float('inf')
    alpha = 2 # phermone factor
    beta = 1024 # distance factor
    gamma = 5 # pixel color factor
    rho = 0.9 # evaporation rate
    rho_min = 0.4
    pheromone_const = 1000
    eps = 0.8 # penalty coefficient for local phermone update
    eps_min = 0.38
    for iter in range(max_iteration):
        # print(f"It's {iter}th loop")
        tabu_list = []
        path_list = []
        for ant in range(ant_num):
            path = []
            transfer_counter = 1
            current_node = starting_position
            c_x, c_y = current_node[0], current_node[1]
            while current_node != ending_position and transfer_counter:
                c_x, c_y = current_node[0], current_node[1]
                neighbor_list = get_neighbors(img, c_x, c_y)
                pixelJ_pair = {}
                q0 = 1 - q0_weight*math.exp(-1/transfer_counter)
                # print(f"Q,Q0:{q},{q0}")
                for neighbor in neighbor_list:
                    if neighbor not in tabu_list:
                        n_x, n_y = neighbor
                        r, g, b = img[n_y][n_x]
                        if ifDiagonal(c_x, c_y, n_x, n_y):
                            cost = 1.4 if (r < 10 and g > 220 and b < 10) else 140.0 
                        else:
                            cost = 1.0 if (r < 10 and g > 220 and b < 10) else 100.0 

                        
                        J = (pheromone_matrix[n_y][n_x]**alpha)*(np.reciprocal(E_distance(n_x,n_y,ending_position[0],ending_position[1]))*beta)*(np.reciprocal(cost)*gamma)
                        if E_distance(n_x,n_y, ending_position[0],ending_position[1]) == 0:
                            current_node = n_x, n_y
                            path.append(current_node)
                            # print("TTTTTTTTTTTTTTARGET FOUND!!!!!!!!!!!")
                            path_list.append(path)
                            break
                        else:
                            pixelJ_pair[(neighbor)] = J
        
                
                if current_node == ending_position:
                    break
                else:
                    picked_neighbor = max(pixelJ_pair.items(), key=lambda item: item[1])
                    # print(f"pixelJ pair:{pixelJ_pair}")
                    if picked_neighbor != starting_position: path.append(picked_neighbor[0])
                    previous_node = current_node
                    coor_x, coor_y = picked_neighbor[0]
                    # print(f"Test coordination: {coor_x}, {coor_y}")
                    local_phermone_update(pheromone_matrix, coor_x,coor_y, eps)
                    current_node = picked_neighbor[0]
    
                    transfer_counter += 1
                    num_pixelJ = len(pixelJ_pair)
                    path.append(current_node)
                    
                    #Conditional Fallback
                    if len(get_neighbors(img, current_node[0], current_node[1])) <= 4:
                        tabu_list.append(current_node)
                        pheromone_matrix[current_node[1]][current_node[0]] = 0
                        current_node = previous_node
                            
                eps = penalty_coef_update(eps, eps_min)
            if current_node == ending_position:
                path_list.append(path)
        global_optimal, global_optimal_length =global_pheromone_update(path_list,
                                                                       pheromone_matrix,
                                                                       rho,rho_min,
                                                                       pheromone_const,
                                                                       max_iteration,iter,
                                                                       global_optimal, global_optimal_length)
    if not global_optimal:
        print("For IACO algorithm, path not found!")
    else:
        img_NIACO, cost_NIACO = print_NIACO(img,starting_position,ending_position,path_color,global_optimal)
    end_time = time.time()
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    print(f"Algorithm execution time: {end_time - start_time:.6f} sec")
    print(f"Peak memory usage: {peak_memory / 1024 / 1024:.2f} MB")
    tracemalloc.stop()
    return img_NIACO, end_time - start_time, peak_memory, cost_NIACO

# Path planning functions
def print_pathDijk(img, starting_position, ending_position, path_color, predecessors):
    cost_Dijkstra = 0
    img_copy = img.copy()
    endpoint_color = (255, 0, 0)
    start_x, start_y = starting_position
    end_x, end_y = ending_position

    path = []
    current = (end_x, end_y)
    while current in predecessors:
        path.append(current)
        current = predecessors[current]
    path.append(starting_position)
    path.reverse()

    prev_x, prev_y = path[0]
    for (px, py) in path:
            pixel_color = img[py, px]
            if ifDiagonal(prev_x, prev_y, px, py):
                if (pixel_color == [255, 255, 255]).all():
                    cost_Dijkstra += 140
                elif (pixel_color == [0, 255, 0]).all():
                    cost_Dijkstra += 1.4
            else:
                if (pixel_color == [255, 255, 255]).all():
                    cost_Dijkstra += 100
                elif (pixel_color == [0, 255, 0]).all():
                    cost_Dijkstra += 1

            img_copy[py][px] = path_color
            img_copy[py + 1][px] = path_color
            img_copy[py - 1][px] = path_color
            img_copy[py][px + 1] = path_color
            img_copy[py][px - 1] = path_color
            img_copy[py + 1][px + 1] = path_color
            img_copy[py + 1][px - 1] = path_color
            img_copy[py - 1][px + 1] = path_color
            img_copy[py - 1][px - 1] = path_color

            prev_x, prev_y = px, py
    
    img_copy[start_y][start_x] = endpoint_color
    img_copy[end_y][end_x] = endpoint_color

    print(f"Total cost for Dijkstra: {cost_Dijkstra}")
    return img_copy, cost_Dijkstra

def print_path(img, starting_position, ending_position, path_color, node_details, dim):
    cost_Astar = 0
    img_copy = img.copy()
    endpoint_color = (255, 0, 0)
    start_x = starting_position[0]
    start_y = starting_position[1]
    end_x = ending_position[0]
    end_y = ending_position[1]
    parent_x = node_details[end_x][end_y][3]
    parent_y = node_details[end_x][end_y][4]
    end_z = int
    if dim == 3:
        end_z = node_details[end_x][end_y][5]

    img_copy[parent_y][parent_x] = path_color
    img_copy[parent_y+1][parent_x+1] = path_color
    img_copy[parent_y+1][parent_x-1] = path_color
    img_copy[parent_y-1][parent_x-1] = path_color
    img_copy[parent_y-1][parent_x+1] = path_color
    img_copy[parent_y + 1][parent_x ] = path_color
    img_copy[parent_y ][parent_x - 1] = path_color
    img_copy[parent_y - 1][parent_x ] = path_color
    img_copy[parent_y ][parent_x + 1] = path_color

    prev_x, prev_y = end_x, end_y
    if dim == 2:
        while True:
            print(f"Current parent: ({parent_x}, {parent_y}), Start: ({start_x}, {start_y})")
            if parent_x == start_x and parent_y == start_y:
                img_copy[start_y][start_x] = endpoint_color
                img_copy[end_y][end_x] = endpoint_color
                print(f"Total cost for A*: {cost_Astar}")
                return img_copy, cost_Astar

            pixel_color = img[parent_y, parent_x]
            
            # Calculate the cost between the current pixel and the previous pixel
            if ifDiagonal(prev_x, prev_y, parent_x, parent_y):
                if (pixel_color == [255, 255, 255]).all():
                    cost_Astar += 140
                elif (pixel_color == [0, 255, 0]).all():
                    cost_Astar += 1.4
            else:
                if (pixel_color == [255, 255, 255]).all():
                    cost_Astar += 100
                elif (pixel_color == [0, 255, 0]).all():
                    cost_Astar += 1

            prev_x, prev_y = parent_x, parent_y

            temp_x = parent_x
            temp_y = parent_y
            parent_x = node_details[temp_x][temp_y][3]
            parent_y = node_details[temp_x][temp_y][4]

            img_copy[parent_y][parent_x] = path_color
            img_copy[parent_y + 1][parent_x + 1] = path_color
            img_copy[parent_y + 1][parent_x - 1] = path_color
            img_copy[parent_y - 1][parent_x - 1] = path_color
            img_copy[parent_y - 1][parent_x + 1] = path_color
            img_copy[parent_y + 1][parent_x] = path_color
            img_copy[parent_y][parent_x - 1] = path_color
            img_copy[parent_y - 1][parent_x] = path_color
            img_copy[parent_y][parent_x + 1] = path_color
    if dim == 3:
         while True:
            print(f"Current parent: ({parent_x}, {parent_y}), Start: ({start_x}, {start_y})")
            if parent_x == start_x and parent_y == start_y:
                img_copy[start_y][start_x] = endpoint_color
                img_copy[end_y][end_x] = endpoint_color
                print(f"Total cost for A*: {cost_Astar}")
                return img_copy, cost_Astar
            
            pixel_z = node_details[parent_x][parent_y][5]
            if ifDiagonal(prev_x, prev_y, parent_x, parent_y):
                cost_Astar += 140 if (pixel_z >= 80).any() else 1.4
            else: cost_Astar += 100 if (pixel_z >= 80).any() else 1

            
            prev_x, prev_y = parent_x, parent_y
           
            temp_x = parent_x
            temp_y = parent_y
            parent_x = node_details[temp_x][temp_y][3]
            parent_y = node_details[temp_x][temp_y][4]

            img_copy[parent_y][parent_x] = path_color
            img_copy[parent_y + 1][parent_x + 1] = path_color
            img_copy[parent_y + 1][parent_x - 1] = path_color
            img_copy[parent_y - 1][parent_x - 1] = path_color
            img_copy[parent_y - 1][parent_x + 1] = path_color
            img_copy[parent_y + 1][parent_x] = path_color
            img_copy[parent_y][parent_x - 1] = path_color
            img_copy[parent_y - 1][parent_x] = path_color
            img_copy[parent_y][parent_x + 1] = path_color

def printRRTstar(img, path_list, path_color):
    cost_RRTstar = 0
    img_copy = img.copy()
    start_x, start_y = path_list[0]
    end_x, end_y = path_list[-1]
    
    img_copy[start_y][start_x] = path_color
    img_copy[end_y][end_x] = path_color

    for i in range(len(path_list) - 1):
        current_node = path_list[i]
        next_node = path_list[i + 1]
        
        current_x, current_y = current_node
        next_x, next_y = next_node
        dx = next_x - current_x
        dy = next_y - current_y
        steps = max(abs(dx), abs(dy))
        
        for step in range(1, steps + 1):
            x = math.ceil(current_x + step * dx / steps)
            y = math.ceil(current_y + step * dy / steps)
            
            # Calculate the cost of the current interpolated pixel
            if ifDiagonal(current_x, current_y, x, y):
                if (img[y, x] == [255, 255, 255]).all():
                    cost_RRTstar += 140
                elif (img[y, x] == [0, 255, 0]).all():
                    cost_RRTstar += 1.4
            else:
                if (img[y, x] == [255, 255, 255]).all():
                    cost_RRTstar += 100
                elif (img[y, x] == [0, 255, 0]).all():
                    cost_RRTstar += 1
            
            # Draw the interpolated pixels
            img_copy[y][x] = path_color
            
            # Paint surrounding pixels to thicken the path
            img_copy[y + 1][x] = path_color
            img_copy[y - 1][x] = path_color
            img_copy[y][x + 1] = path_color
            img_copy[y][x - 1] = path_color
            img_copy[y + 1][x + 1] = path_color
            img_copy[y + 1][x - 1] = path_color
            img_copy[y - 1][x + 1] = path_color
            img_copy[y - 1][x - 1] = path_color
    
    img_copy[end_y][end_x] = path_color
    
    print(f"Total cost for RRT*: {cost_RRTstar}")
    return img_copy, cost_RRTstar

def print_NIACO(img,starting_position,ending_position,path_color,global_optimal):
    cost_global = 0
    img_copy = img.copy()

    prev_x, prev_y = global_optimal[0]

    # Traversing the global optimal path
    for (px, py) in global_optimal:
        pixel_color = img[py, px]

        if ifDiagonal(prev_x, prev_y, px, py):
            if (pixel_color == [255, 255, 255]).all():
                cost_global += 140
            elif (pixel_color == [0, 255, 0]).all():
                cost_global += 1.4
        else:
            if (pixel_color == [255, 255, 255]).all():
                cost_global += 100
            elif (pixel_color == [0, 255, 0]).all():
                cost_global += 1

        img_copy[py][px] = path_color
        img_copy[py + 1][px] = path_color
        img_copy[py - 1][px] = path_color
        img_copy[py][px + 1] = path_color
        img_copy[py][px - 1] = path_color
        img_copy[py + 1][px + 1] = path_color
        img_copy[py + 1][px - 1] = path_color
        img_copy[py - 1][px + 1] = path_color
        img_copy[py - 1][px - 1] = path_color

        prev_x, prev_y = px, py

    return img_copy, cost_global




#from 'get_neighbors' to 'global_pheromone_update' are IACO related tool functions 
def get_neighbors(img, x, y):
    all_neighbors = [(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx != 0 or dy != 0)]
    available_list = []
    for neighbor in all_neighbors:
        # First check whether it is within the image range
        if exists(img, neighbor[0], neighbor[1], 1024, 1024) and is_traversable(img, neighbor[0], neighbor[1]):
            available_list.append(neighbor)
    return available_list

def local_phermone_update(phermone_matrix, coor_x, coor_y, eps):
    phermone_matrix[coor_y][coor_x] = eps * phermone_matrix[coor_y][coor_x]

def penalty_coef_update(eps, eps_min):
    if eps <= eps_min:
        eps = eps_min
    else:
        eps = 0.5 * eps + 0.5*((1 - eps)**2)
    
    return eps

def calculate_path_length(path):
    if not path:
        length = 0
        print("Route not found yet!")
        return length
    else:
        length = 0
        for i in range(len(path) - 1):
            length += E_distance(path[i][0], path[i][1], path[i+1][0], path[i+1][1])
        return length
    
def global_pheromone_update(path_list, pheromone_matrix, rho,rho_min, pheromone_const,max_iterations, iter, global_optimal, global_optimal_length):
    if rho <= rho_min:
        rho = rho_min
    else:
        rho = 0.9 * rho + 0.1 * ((1-rho)**2)

    shortest_path = []
    shortest_length = float('inf')
    
    for path in path_list:
        path_length = calculate_path_length(path)
        if path_length < shortest_length and path_length != 0:
            shortest_length = path_length
            shortest_path = path
        if path_length < global_optimal_length and path_length != 0:
            global_optimal = path
            global_optimal_length = path_length
    if shortest_path:
        for i in range(len(shortest_path) - 1):
            coorx = shortest_path[i][0]
            coory = shortest_path[i][1]
            pheromone_matrix[coory][coorx] = 0.9 * pheromone_matrix[coory][coorx] + (1 - 0.9)*(((3*max_iterations+pheromone_const)/shortest_length))-((3*iter)/shortest_length)

    return global_optimal, global_optimal_length



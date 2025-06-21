import math
import numpy as np
import time
import heapq
import random
import tracemalloc
from Tools import exists, is_traversable, is_destination, ifDiagonal, E_distance, heuristics3D, get_neighbors

t1 = time.perf_counter()

# 3D path planning implementations
def findbyAstar3D(img, img_gray,img_combined, width, height, starting_position, ending_position, path_color):
    tracemalloc.start()
    start_time = time.time()

    # if the user chose the starting and ending position to be the same
    if is_destination(img, starting_position[0], starting_position[1], ending_position):
        print("The starting and ending position you have chosen is the same.")
        print("Exiting program.")
        exit()
    
    # f, g, h, parent x,y,height
    node_details = np.full((width, height, 6), -1)

    # create a 2D-list that marks all nodes that have been evaluated by 1, 0 if not
    closed_list = np.zeros((width, height))

    # each index will contain: [f, [x, y]]
    open_list = []
    open_list.append([0, [starting_position[0], starting_position[1]]])
    while(len(open_list)!= 0):
        if len(open_list) == 1:
            current = open_list[0]
        else:
            current = min(open_list, key=lambda x: (x[0], node_details[x[1][0], x[1][1], 2]))
        
        cur_x, cur_y = current[1]
        open_list.remove(current)
        closed_list[cur_x, cur_y] = 1

        if is_destination(img, cur_x, cur_y, ending_position):
            print("Path was successfully found.")
            img_Astar, cost_Astar = print_path(img_combined, starting_position, ending_position, path_color, node_details, dim=3)
            end_time = time.time()
            current, peak = tracemalloc.get_traced_memory()
            print(f"Algorithm execution time: {end_time - start_time} sec ")
            print(f"Peak memory usage: {peak / 1024 / 1024:.2f}MB")
            tracemalloc.stop()
            return img_Astar, end_time - start_time, peak, cost_Astar

        neighbor_list = get_neighbors(img, cur_x, cur_y)

        for neighbor in neighbor_list:
            n_x, n_y = neighbor
            if exists(img, n_x, n_y, width, height) and is_traversable(img, n_x, n_y) and closed_list[n_x, n_y]==0:
                n_z = img_gray[n_y, n_x]
                if current[1] == starting_position:
                    if ifDiagonal(cur_x, cur_y, n_x, n_y):
                        g = 140 if (n_z >= 80).any() else 1.4
                    else:
                        g = 100 if (n_z >= 80).any() else 1
                else:
                    base_g = node_details[cur_x, cur_y, 1]
                    temp_g = 0
                    if ifDiagonal(cur_x, cur_y, n_x, n_y):
                        temp_g = 140 if (n_z >= 80).any() else 1.4
                    else:
                        temp_g = 100 if (n_z >= 80).any() else 1
                    g = base_g + temp_g
                
                h = heuristics3D(img,img_gray, n_x, n_y, ending_position)
                f = g + h
                update = True
                in_open = False
                index = -1
                count = 0

                for count, node in enumerate(open_list):
                    node_f, (node_x, node_y) = node
                    if n_x == node_x and n_y == node_y:
                        in_open = True
                        index = count
                        if f >= node_f:
                            update = False

                if update:
                    node_details[n_x, n_y] = [f, g, h, cur_x, cur_y, n_z]
                    if not in_open:
                        open_list.append([f, [n_x, n_y]])
                    else:
                        open_list[index][0] = f
    print("Path was not found. Exiting program.")
    exit()


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

def findbyDijkstra3D(img, img_gray,img_combined, width, height, starting_position, ending_position, path_color):
    tracemalloc.start()
    start_time = time.time()

    distance = np.full((height, width), np.inf)
    distance[starting_position[1], starting_position[0]] = 0

    pq = []
    heapq.heappush(pq, (0, starting_position))  # (distance, (x, y)
    predecessors = {}
    while pq:
        cur_distance, current = heapq.heappop(pq)
        cur_x, cur_y = current

        # If we reached the ending position, break the loop
        if current == (ending_position[0], ending_position[1]):
            print("Path was successfully found.")
            break

        neighbor_list = get_neighbors(img, cur_x, cur_y)
        for neighbor in neighbor_list:
            n_x, n_y = neighbor
            if exists(img, n_x, n_y, width, height) and is_traversable(img, n_x, n_y):
                n_z = img_gray[n_y][n_x]
                if ifDiagonal(cur_x, cur_y, n_x, n_y):
                    n_d = 1.4 if n_z < 80 else 140
                else:
                    n_d = 1 if n_z < 80 else 100
                tentative_distance = cur_distance + n_d
                if tentative_distance < distance[n_y, n_x]:
                    distance[n_y, n_x] = tentative_distance
                    predecessors[(n_x, n_y)] = (cur_x, cur_y)  # Record predecessor
                    heapq.heappush(pq, (tentative_distance, (n_x, n_y)))
    
    img_dij, cost_Dijkstra = print_pathDijk3D(img_combined, img_gray ,starting_position,ending_position,path_color,predecessors)
    end_time = time.time()
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    print(f"Algorithm execution time: {end_time - start_time:.6f} sec")
    print(f"Peak memory usage: {peak_memory / 1024 / 1024:.2f} MB")
    tracemalloc.stop()
    return img_dij, end_time - start_time, peak_memory, cost_Dijkstra

def findbyrrtconnect(img, img_gray, img_combined, width, height, starting_position, ending_position, path_color):
    tracemalloc.start()
    start_time = time.time()
    max_iterations = np.inf
    forward_tree = [starting_position]
    backward_tree = [ending_position]
    predecessors = {}
    step_size = 3.0

    while True:
        if len(forward_tree) > len(backward_tree):
            forward_tree, backward_tree = backward_tree, forward_tree
            sp_temp, ep_temp = ending_position, starting_position  # 仅用于路径连接
        else:
            sp_temp, ep_temp = starting_position, ending_position


        random_node = (np.random.randint(0, width), np.random.randint(0, height))
        if not is_traversable(img, random_node[0], random_node[1]):
            continue

        # Find closest node in the forward tree
        dlist = [(node[0] - random_node[0])**2 + (node[1] - random_node[1])**2 for node in forward_tree]
        closest_node = tuple(forward_tree[dlist.index(min(dlist))])
        distance = rrtcon_cal_distance(closest_node, random_node)
        new_node = rrtcon_steer(img, img_gray, closest_node, random_node, distance, step_size)

        if new_node is None:
            continue

        forward_tree.append(new_node)
        predecessors[new_node] = closest_node
        print(f"Adding to predecessors: {new_node} -> {closest_node}")
        # Check backward tree for connection
        elist = [(node[0] - new_node[0])**2 + (node[1] - new_node[1])**2 for node in backward_tree]
        closest_end = tuple(backward_tree[elist.index(min(elist))])
        distance_end = rrtcon_cal_distance(closest_end, new_node)
        new_end = rrtcon_steer(img, img_gray, closest_end, new_node, distance_end, step_size)
        print(f"New end: {new_end}")
        if new_end is None:
            continue

        backward_tree.append(new_end)
        predecessors[new_end] = closest_end
        print(f"Adding to predecessors: {new_end} -> {closest_end}")
        print(f"nodes distance{rrtcon_cal_distance(new_end, new_node)}")
        if rrtcon_cal_distance(new_end, new_node) <= step_size:
            path = rrt_construct_path(new_node, new_end, sp_temp, ep_temp ,predecessors)
            img_rrt_c, cost_rrt_c = print_pathRRT3D(img_combined, img_gray, sp_temp, ep_temp, path, path_color)
            end_time = time.time()
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            print(f"Algorithm execution time: {end_time - start_time:.6f} sec")
            print(f"Peak memory usage: {peak_memory / 1024 / 1024:.2f} MB")
            tracemalloc.stop()
            return img_rrt_c, end_time - start_time, peak_memory, cost_rrt_c

def findbyNIACO3D(img, img_gray, img_combined, width, height, starting_position, ending_position, path_color):
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
    beta = 1000 # distance factor
    gamma = 5 # pixel color factor
    rho = 0.9 # evaporation rate
    rho_min = 0.4
    pheromone_const = 1000
    eps = 0.8 # penalty coefficient for local phermone update
    eps_min = 0.38
    for iter in range(max_iteration):
        print(f"It's {iter}th loop")
        tabu_list = []
        path_list = []
        for ant in range(ant_num):
            path = []
            transfer_counter = 1
            current_node = starting_position
            c_x, c_y = current_node[0], current_node[1]
            print(f"Current position {c_x}, {c_y}")
            #tabu_list.append(current_node)
            while current_node != ending_position and transfer_counter:
                c_x, c_y = current_node[0], current_node[1]
                neighbor_list = get_neighbors(img, c_x, c_y)
                pixelJ_pair = {}
                q0 = 1 - q0_weight*math.exp(-1/transfer_counter)
                print(f"Q,Q0:{q},{q0}")
                for neighbor in neighbor_list:
                    if neighbor not in tabu_list:
                        n_x, n_y = neighbor
                        n_z = img_gray[n_y][n_x]
                        if ifDiagonal(c_x, c_y, n_x, n_y):
                            cost = 1.4 if (n_z <= 80) else 140.0 
                        else:
                            cost = 1.0 if (n_z <= 80) else 100.0 

                        J = (pheromone_matrix[n_y][n_x]**alpha)*(np.reciprocal(E_distance(n_x,n_y,ending_position[0],ending_position[1]))*beta)*(np.reciprocal(cost)*gamma)
                        
                        if E_distance(n_x,n_y, ending_position[0],ending_position[1]) == 0:
                            current_node = n_x, n_y
                            path.append(current_node)
                            print("TTTTTTTTTTTTTTARGET FOUND!!!!!!!!!!!")
                            path_list.append(path)
                            break
                        else:
                            pixelJ_pair[(neighbor)] = J
                            
                if current_node == ending_position:
                    break
                else:
                        picked_neighbor = max(pixelJ_pair.items(), key=lambda item: item[1])
                        print(f"pixelJ pair:{pixelJ_pair}")
                        if picked_neighbor != starting_position: path.append(picked_neighbor[0])
                        previous_node = current_node
                        coor_x, coor_y = picked_neighbor[0]
                        print(f"Test coordination: {coor_x}, {coor_y}")
                        local_phermone_update(pheromone_matrix, coor_x,coor_y, eps)
                        current_node = picked_neighbor[0]
                        if current_node == ending_position:
                            print("TTTTTTTTTTTTTTARGET FOUND!!!!!!!!!!!")
                        transfer_counter += 1
                        num_pixelJ = len(pixelJ_pair)
                        path.append(current_node)
                        print(f"Transfer counter: {transfer_counter}")
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
        img_IACO, cost_IACO = print_pathIACO3D(img_combined, img_gray, starting_position,ending_position,path_color,global_optimal)
    end_time = time.time()
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    print(f"Algorithm execution time: {end_time - start_time:.6f} sec")
    print(f"Peak memory usage: {peak_memory / 1024 / 1024:.2f} MB")
    tracemalloc.stop()
    return img_IACO, end_time - start_time, peak_memory, cost_IACO 
   
# 3D path printing
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

def print_pathDijk3D(img_combined, img_gray ,starting_position, ending_position, path_color, predecessors):
    img_copy = img_combined.copy()
    cost_dijk3D = 0 
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
        pz = img_gray[py][px]
        if ifDiagonal(prev_x,prev_y,px,py):
            cost_dijk3D += 1.4 if pz < 80 else 140
        else:
            cost_dijk3D +=1  if pz < 80 else 100
        img_copy[py][px] = path_color
        img_copy[py + 1][px] = path_color
        img_copy[py - 1][px] = path_color
        img_copy[py][px + 1] = path_color
        img_copy[py][px - 1] = path_color
        img_copy[py + 1][px + 1] = path_color
        img_copy[py + 1][px - 1] = path_color
        img_copy[py - 1][px + 1] = path_color
        img_copy[py - 1][px - 1] = path_color

        prev_x , prev_y = px, py
    img_copy[start_y][start_x] = endpoint_color
    img_copy[end_y][end_x] = endpoint_color

    print(f"Total cost for Dijkstra: {cost_dijk3D}")
    return img_copy, cost_dijk3D

def print_pathRRT3D(img_combined, img_gray, starting_position, ending_position, path, path_color):
    print("Start printing path")
    img_copy = img_combined.copy()
    cost_rrt_c = 0
    endpoint_color = (255, 0, 0)  
    start_x, start_y = starting_position
    end_x, end_y = ending_position

    img_copy[start_y][start_x] = endpoint_color
    img_copy[end_y][end_x] = endpoint_color

    prev_x, prev_y = path[0]

    for (px, py) in path[1:]:
        distance = np.hypot(px - prev_x, py - prev_y)
        num_steps = int(distance)  # The number of steps is set to the integer part of the Euclidean distance between two nodes.

        for step in range(1, num_steps + 1):
            inter_x = int(prev_x + (px - prev_x) * (step / num_steps))
            inter_y = int(prev_y + (py - prev_y) * (step / num_steps))
            inter_z = img_gray[inter_y][inter_x]

            if ifDiagonal(prev_x, prev_y, inter_x, inter_y):
                cost_rrt_c += 1.4 if inter_z < 80 else 140
            else:
                cost_rrt_c += 1 if inter_z < 80 else 100

            img_copy[inter_y][inter_x] = path_color
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if 0 <= inter_x + dx < img_copy.shape[1] and 0 <= inter_y + dy < img_copy.shape[0]:
                        img_copy[inter_y + dy][inter_x + dx] = path_color

        prev_x, prev_y = px, py

    print(f"Total cost for RRT-Connect: {cost_rrt_c}")
    return img_copy, cost_rrt_c

def print_pathIACO3D(img_combined,img_gray, starting_position,ending_position,path_color,global_optimal):
    cost_global = 0
    img_copy = img_combined.copy()

    start_x, start_y = starting_position
    end_x, end_y = ending_position
    prev_x, prev_y = global_optimal[0]
    for (px, py) in global_optimal:
        pz = img_gray[py, px]
        if ifDiagonal(prev_x, prev_y, px, py):
            if (pz > 80).all():  
                cost_global += 140
            elif (pz <= 80).all():
                cost_global += 1.4
        else:
            if (pz > 80).all():
                cost_global += 100
            elif (pz <= 80).all():
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
    img_copy[start_y][start_x] = path_color
    img_copy[end_y][end_x] = path_color
    
    return img_copy, cost_global

# RRT-Connect related tool functions
def is_path_clear(img, start_node, end_node, step_size=1):
    "Check if there are black pixels (inaccessible areas) on the path from the start point to the end point"
    direction = np.array(end_node) - np.array(start_node)
    norm = np.linalg.norm(direction)
    num_steps = int(norm // step_size)

    for step in range(1, num_steps + 1):
        intermediate_point = start_node + direction * (step / num_steps)
        x, y = int(intermediate_point[0]), int(intermediate_point[1])
        
        if img[y, x] == 0:
            return False

    return True

def rrtcon_cal_distance(from_node, to_node):
    diff_x = to_node[0] - from_node[0]
    diff_y = to_node[1] - from_node[1]
    e_dis = math.hypot(diff_x, diff_y)
    return e_dis

def rrtcon_steer(img, img_gray ,closest_node,random_node, distance,step_size):
    # dynamic step_size
    closest_z = img_gray[closest_node[1]][closest_node[0]]
    random_z = img_gray[random_node[1]][random_node[0]]
    gradient = random_z - closest_z
    #step_size /= 2 if gradient > 10 else step_size
    
    if distance < step_size: new_node = tuple(random_node)
    else:
        direction = np.array(random_node) - np.array(closest_node)
        norm = np.linalg.norm(direction)
        step = (direction/norm) * step_size
        new_node = (int(closest_node[0] + step[0]), int(closest_node[1] + step[1]))
        if is_traversable(img, new_node[0], new_node[1]) and is_path_clear(img_gray, closest_node, new_node, step_size = 1):
            return new_node
        return

def rrt_construct_path(new_node, new_end, starting_position, ending_position ,predecessors):
    path = []
    
    # Backtrack from new_node to the starting point
    current = new_node
    iteration_count = 0
    while tuple(current) in predecessors:
        if current == starting_position:
            break
        path.append(current)
        current = predecessors[tuple(current)]
        
    path.reverse()  # Reverse to get the correct order

    # Adding the connection section
    path.append(new_end)

    # Backtrack from end_node to the end point
    current = new_end
    iteration_count = 0
    while tuple(current) in predecessors:
        if current == ending_position:
            break
        path.append(current)
        current = predecessors[tuple(current)]
        
    print("Path found")
    return path


#NIACO related utils functions
#from 'get_neighbors' to 'global_pheromone_update' are NIACO related tool functions 
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
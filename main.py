# cannot traverse/black = values below "[5, 5, 5]"
# can traverse/white = values above "[250, 250, 250]" 

import time
import cv2
import csv
import os
import json 
import numpy as np
import matplotlib.pyplot as plt

from find2D import findbyAStar
from find2D import findbyDijkstra
from find2D import findbyrrtstar
from find2D import findbyNIACO

from find3D import findbyAstar3D
from find3D import findbyDijkstra3D
from find3D import findbyrrtconnect
from find3D import findbyNIACO3D


t1=time.perf_counter()
flag = 0
# img = cv2.imread("./data/70933/70933_weighted.png")
img = cv2.imread("./data/70934/70934_mask.png")
# img = cv2.imread("./data/UAV_data/UAV_data_weighted.png")
# img = cv2.imread("./data/UAV_data/UAV_mask.png")
# img_elevation = cv2.imread("./data/UAV_data/UAV_graymask.png")
img_elevation = cv2.imread("./data/70934/70934_graymask.png")
img_combined = cv2.imread("./data/70934/70934_combined.png")
# img_combined = cv2.imread("./data/UAV_data/UAV_combined.png")
#img = cv2.imread("70933_mask.png")
#img = cv2.imread("70933_sat.jpg")

print(img)

width = int
height = int
x = int
y = int
window_name = "path finding"
starting_position = list()
ending_position = list()
path_color = (0, 0, 255)

def main():

    global img
    global img_elevation
    global img_combined
    global width, height
    global window_name
    
    dimensions = img.shape
    height = dimensions[0]
    width = dimensions[1]
    #gray_img_elevation = cv2.cvtColor(img_elevation, cv2.COLOR_BGR2GRAY)
    #print(f"gray img test {gray_img_elevation[491][381]}")
    #displaying the image in a window for a user to choose a starting position
    print("Choose a valid starting position (white pixel) by double left clicking on the image.")
    print("Click any key to exit out of this program at any time.\n")

    cv2.imshow(window_name, img)
    
    cv2.setMouseCallback(window_name, mouse_events)
    cv2.waitKey(0)


def mouse_events(event, x, y, flags, param):
    global flag

    global img
    global starting_position, ending_position
    global path_color

    if event == cv2.EVENT_LBUTTONDBLCLK:
        # need to check if rgb values correspond to the color white
        r = img[y][x][0]
        g = img[y][x][1]
        b = img[y][x][2]
        color_flag = 0
        if r >= 250 and g >= 250 and b >= 250:
            color_flag = 1
        if  r < 10 and g > 220 and b < 10:
                color_flag = 1
        else:
            print("You have chosen an invalid position. Choose a valid one.")

        # getting starting position
        if flag == 0 and color_flag == 1:
            cv2.imshow(window_name, img)
            print("The starting position you chose was: (%d, %d)" % (x, y))
            starting_position.append(x)
            starting_position.append(y)
            flag += 1
        # getting ending position
        elif flag == 1 and color_flag == 1:
            cv2.imshow(window_name, img)
            print("The ending position you chose was: (%d, %d)" % (x, y))
            ending_position.append(x)
            ending_position.append(y)

            # calling function to find path
            # find_path()
            find_path3D()
            flag += 1
            
def find_path():
    global img
    global width, height
    global x, y
    global starting_position, ending_position
    colors = {
        'A*': (255, 0, 0),        # blue
        'Dijkstra': (0, 215, 255),  # gold
        'RRT*': (0, 0, 255),      # red
        'NIACO': (255, 215, 0) ,   # light blue   
    }
    img_combined = img.copy()
    
    print("Finding path using A*...")
    img_a_star, time_1, memory_1, cost_Astar = findbyAStar(img, width, height, starting_position, ending_position, colors["A*"])

    print("Finding path using Dijkstra...")
    img_dijkstra, time_2, memory_2, cost_Dijkstra = findbyDijkstra(img, width, height, starting_position, ending_position, colors["Dijkstra"])

    print("Finding path using RRT*...")
    img_rrt_star, time_3, memory_3, cost_RRTStar = findbyrrtstar(img, width, height, starting_position, ending_position, colors["RRT*"])
    
    print("Finding path using NIACO...")
    img_niaco, time_4, memory_4, cost_NIACO = findbyNIACO(img, width, height, starting_position, ending_position, colors["NIACO"])

    


    
    # combine all the path into img_combined
    for y in range(height):
        for x in range(width):
            if (img_a_star[y, x] == colors["A*"]).all():
                img_combined[y, x] = colors["A*"]
            elif (img_dijkstra[y, x] == colors["Dijkstra"]).all():
                img_combined[y, x] = colors["Dijkstra"]
            if (img_rrt_star[y, x] == colors["RRT*"]).all():
                img_combined[y, x] = colors["RRT*"]
            if (img_niaco[y, x] == colors["NIACO"]).all():
                img_combined[y, x] = colors["NIACO"]

    cv2.imshow('Path Results', img_combined)

    print(f"Cost of A* algorithm: {cost_Astar}")
    print(f"Cost of Dijkstra Algorithm: {cost_Dijkstra}")
    print(f"Cost of RRT* Algorithm: {cost_RRTStar}")
    print(f"Cost of NIACO Algorithm: {cost_NIACO}")
    print(f"Time used for A*: {time_1}, Memory used: {memory_1 / 1024 / 1024:.2f} MB")
    print(f"Time used for Dijkstra: {time_2}, Memory used: {memory_2 / 1024 / 1024:.2f} MB")
    print(f"Time used for RRT*: {time_3}, Memory used: {memory_3 / 1024 / 1024:.2f} MB")
    print(f"Time used for NIACO: {time_4}, Memory used: {memory_4 / 1024 / 1024:.2f} MB")

    algorithms = ['A*', 'Dijkstra', 'RRT*','IACO']
    times = [time_1, time_2, time_3,time_4]  
    memories = [memory_1 / 1024 / 1024, memory_2 / 1024 / 1024, memory_3 / 1024 / 1024, memory_4 / 1024 / 1024]
    costs = [cost_Astar, cost_Dijkstra, cost_RRTStar, cost_NIACO]

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Algorithms')
    ax1.set_ylabel('Time (s)', color='tab:blue')
    ax1.plot(algorithms, times, color='tab:blue', marker='o', label='Time (s)')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Memory (MB)', color='tab:red')
    ax2.bar(algorithms, memories, color='tab:red', alpha=0.6, label='Memory (MB)')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel('Cost', color='tab:green')
    ax3.plot(algorithms, costs, color='tab:green', marker='x', label='Cost', linestyle='--')
    ax3.tick_params(axis='y', labelcolor='tab:green')

    fig.tight_layout()
    plt.title('Comparison of Time, Memory Usage, and Cost Among Algorithms')


    plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    csv_file = '2D path_planning_results.csv'
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        
        if not file_exists:
            writer.writerow([
                'Start_X', 'Start_Y', 'End_X', 'End_Y',
                'A*_Cost', 'A*_Time(s)', 'A*_Memory(MB)',
                'Dijkstra_Cost', 'Dijkstra_Time(s)', 'Dijkstra_Memory(MB)',
                'RRT*_Cost', 'RRT*_Time(s)', 'RRT*_Memory(MB)',
                'NIACO_Cost', 'NIACO_Time(s)', 'NIACO_Memory(MB)'
            ])

        writer.writerow([
            starting_position[0], starting_position[1], ending_position[0], ending_position[1],
            cost_Astar, time_1, memory_1 / 1024 / 1024,
            cost_Dijkstra, time_2, memory_2 / 1024 / 1024,
            cost_RRTStar, time_3, memory_3 / 1024 / 1024,
            cost_NIACO, time_4, memory_4 / 1024 / 1024
        ])
    save_file(img_combined)

def find_path3D():
    global img
    global img_elevation
    global width, height
    global x, y
    global starting_position, ending_position
    colors = {
        '3D A*': (255, 0, 0),        # blue
        '3D Dijkstra': (0, 215, 255),  # gold
        'RRT-Connect': (0, 0, 255),      # red
        'NIACO 3D': (255, 215, 0) ,   # light blue
    }
    img_combined_copy = img_combined.copy()
    # convert the original elevation image into gray image
    img_gray = cv2.cvtColor(img_elevation, cv2.COLOR_BGR2GRAY)

    print("finding path with rrt-connect...")
    img_rrtc, time_3, memory_3, cost_rrtc = findbyrrtconnect(img, img_gray, img_combined,width, height, starting_position, ending_position, colors["RRT-Connect"])
    
    print("finding path with 3D A* algorithm...")
    img_astar, time_1, memory_1, cost_Astar = findbyAstar3D(img, img_gray, img_combined,width, height, starting_position, ending_position, colors["3D A*"])
    
    print("finding path with 3D Dijkstra...")
    img_dijk, time_2, memory_2, cost_dijk = findbyDijkstra3D(img, img_gray, img_combined,width, height, starting_position, ending_position, colors["3D Dijkstra"])
    
    print("finding path with IACO3D")
    img_niaco, time_4, memory_4, cost_niaco = findbyNIACO3D(img, img_gray, img_combined,width, height, starting_position, ending_position, colors["NIACO 3D"])

    for y in range(height):
        for x in range(width):
            if (img_rrtc[y, x] == colors["RRT-Connect"]).all():
                img_combined_copy[y, x] = colors["RRT-Connect"]
            elif (img_astar[y, x] == colors["3D A*"]).all():
                img_combined_copy[y, x] = colors["3D A*"]
            elif (img_dijk[y, x] == colors["3D Dijkstra"]).all():
                img_combined_copy[y, x] = colors["3D Dijkstra"]
            elif (img_niaco[y, x] == colors["NIACO 3D"]).all():
                img_combined_copy[y, x] = colors["NIACO 3D"]

    cv2.imshow('Path Results', img_combined_copy)
    
    print(f"Cost of A* algorithm: {cost_Astar}")
    print(f"Cost of Dijkstra Algorithm: {cost_dijk}")
    print(f"Cost of RRT* Algorithm: {cost_rrtc}")
    print(f"Cost of NIACO 3D Algorithm: {cost_niaco}")
    print(f"Time used for 3D A*: {time_1}, Memory used: {memory_1 / 1024 / 1024:.2f} MB")
    print(f"Time used for 3D Dijkstra: {time_2}, Memory used: {memory_2 / 1024 / 1024:.2f} MB")
    print(f"Time used for RRT-connect: {time_3}, Memory used: {memory_3 / 1024 / 1024:.2f} MB")
    print(f"Time used for 3D NIACO : {time_4}, Memory used: {memory_4 / 1024 / 1024:.2f} MB")
    
    algorithms = ['A*', 'Dijkstra', 'RRT-Connect', 'NIACO']
    times = [time_1, time_2, time_3, time_4]
    memories = [memory_1 / 1024 / 1024, memory_2 / 1024 / 1024, memory_3 / 1024 / 1024, memory_4 / 1024 / 1024]
    costs = [cost_Astar, cost_dijk, cost_rrtc, cost_niaco]

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Algorithms')
    ax1.set_ylabel('Time (s)', color='tab:blue')
    ax1.plot(algorithms, times, color='tab:blue', marker='o', label='Time (s)')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Memory (MB)', color='tab:red')
    ax2.bar(algorithms, memories, color='tab:red', alpha=0.6, label='Memory (MB)')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel('Cost', color='tab:green')
    ax3.plot(algorithms, costs, color='tab:green', marker='x', label='Cost', linestyle='--')
    ax3.tick_params(axis='y', labelcolor='tab:green')
    
    fig.tight_layout()
    plt.title('Comparison of Time, Memory Usage, and Cost Among Algorithms')

    plt.savefig('3Dalgorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    csv_file = '3Dpath_planning_results.csv'
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                'Start_X', 'Start_Y', 'End_X', 'End_Y',
                'A*_Cost', 'A*_Time(s)', 'A*_Memory(MB)',
                'Dijkstra_Cost', 'Dijkstra_Time(s)', 'Dijkstra_Memory(MB)',
                'RRT-Connect_Cost', 'RRT-Connect_Time(s)', 'RRT-Connect_Memory(MB)',
                'NIACO_Cost', 'NIACO_Time(s)', 'NIACO_Memory(MB)'
            ])

        writer.writerow([
            starting_position[0], starting_position[1], ending_position[0], ending_position[1],
            cost_Astar, time_1, memory_1 / 1024 / 1024,
            cost_dijk, time_2, memory_2 / 1024 / 1024,
            cost_rrtc, time_3, memory_3 / 1024 / 1024,
            cost_niaco, time_4, memory_4 / 1024 / 1024
        ])
    save_file(img_combined_copy)
    


def save_file(img):
    # writing the final image as user inputted filename
    filename = input("\nEnter a filename for the image to save as: ")
    try:
        cv2.imwrite(filename, img)
    except:
        print("The filename you entered is invalid. Exiting program.")
        exit()

    print("The image was successfully saved. Exiting program.")
    exit()
    

if __name__ == "__main__":

    main()


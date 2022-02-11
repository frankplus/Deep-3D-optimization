import csv
import re
import matplotlib.pyplot as plt

# generate actual series of positions from the subjective experiment logs

def open_movement_log(userid, displayid):
    with open(f"data/Log movimento/fileslog_{displayid}_{userid}.csv", newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        first = True
        data = list()
        for row in reader:
            if first: 
                first = False
                continue
            s = ''.join(row)
            pattern = '([0-9]+),"\((-?\d*\.?\d*),(-?\d*\.?\d*),(-?\d*\.?\d*)\)","\((-?\d*\.?\d*),(-?\d*\.?\d*),(-?\d*\.?\d*),(-?\d*\.?\d*)\)",(-?\d*\.?\d*),(U\d+),([a-zA-Z0-9_]*)'
            groups = re.match(pattern, s).groups()
            parsed = {'Timestamp': int(groups[0]), 
                    'Position': list(map(lambda x: float(x), groups[1:4])), 
                    'Rotation': list(map(lambda x: float(x), groups[4:8])), 
                    'fps': float(groups[8]), 
                    'UserID': groups[9],
                    'DisplayID': groups[10]}
            
            # shift y coordinate by -1 because of incompatibility between logs and trained model
            parsed['Position'][1] -=1 

            data.append(parsed)
        return data

def plot(points):
    print(f"{len(points)} positions")
    x_coords = [p[0] for p in points]
    y_coords = [p[2] for p in points]
    plt.scatter(x_coords, y_coords, 1)
    plt.show()

def preprocess_positions(delta_t = 100):
    movement_log = open_movement_log("U081220", "Drago_01")
    positions = list()
    prev_timestamp = movement_log[0]["Timestamp"]
    for log in movement_log:
        if log["Timestamp"] - prev_timestamp > delta_t:
            positions.append(log["Position"])
            prev_timestamp = log["Timestamp"]
    return positions

def save_path(points, filename):
    with open(filename, 'w') as f:
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]} 0 0 0\n")

positions = preprocess_positions()
plot(positions)

save_path(positions, "positions.txt")
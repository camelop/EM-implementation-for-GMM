import csv
import os, datetime
import numpy as np
input_dir = "data"
output_dir = "results"
class Data:
    def __init__(self, l=None, r=None):
        if l is None:
            self.x = []
            self.y = []
        else:
            self.x = l.x+r.x
            self.y = l.y+r.y

    def append(self, x, y):
        self.x.append(x)
        self.y.append(y)

    def output(self, des=None):
        if des is None:
            index = 0
            des = os.path.join(output_dir, str(datetime.date.today())+"-"+str(index)+".csv")
            while(os.path.exists(des)):
                index += 1
                des = os.path.join(output_dir, str(datetime.date.today())+"-"+str(index)+".csv")
                
        with open(des, "w", newline="") as csvFile:
            dataWriter = csv.writer(csvFile)
            dataWriter.writerow(["id", "classes"])
            for i in range(len(self.y)):
                dataWriter.writerow([str(i), str(self.y[i])])
    def nx(self):
        return np.array(self.x)
    def ny(self):
        return np.array(self.y)

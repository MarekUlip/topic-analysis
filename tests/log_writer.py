import datetime
import csv
import os
import matplotlib.pyplot as plt
from tests.ModelType import ModelType

class LogWriter:
    def __init__(self, log_file_path):
        self.path = log_file_path
        self.logs = ["*****************\n"]

    def add_log(self, log):
        log = str(datetime.datetime.now()) + ": "+log+"\n"
        print(log)
        self.logs.append(log)
        if len(self.logs) > 10:
            self.append_to_logfile()

    def append_to_logfile(self):
        with open(self.path, "a+") as f:
            for item in self.logs:
                f.write(item)
            self.logs.clear()

    def end_logging(self):
        self.append_to_logfile()

    def write_2D_list(self, list_name, statistics):
        filename = os.getcwd() + list_name + ".csv"
        print(filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, mode='w+', newline='') as list_file:
            list_writer = csv.writer(list_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for item in statistics:
                list_writer.writerow(item)

    def add_to_plot(self, line_name, points):
        points = [x * 100 for x in points]
        plt.plot(points, label=line_name)

    def draw_plot(self, plot_name, file_name, num_of_tests):
        plt.axis([0, num_of_tests, 0, 100])
        plt.title(plot_name)
        plt.xlabel("Číslo testu")
        plt.ylabel("Přesnost (%)")
        plt.legend()
        path = os.getcwd()+file_name+".png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.clf()

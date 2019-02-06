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
        """
        Adds on line into log file. Note that this method does not write to file. Use append_to_file to perform file writting
        :param log: Text to be appended
        """
        log = str(datetime.datetime.now()) + ": "+log+"\n"
        print(log)
        self.logs.append(log)
        if len(self.logs) > 10:
            self.append_to_logfile()

    def append_to_logfile(self):
        """
        Appends all items in log memory into the log file and clears outputed lines from memory.
        """
        with open(self.path, "a+") as f:
            for item in self.logs:
                f.write(item)
            self.logs.clear()

    def end_logging(self):
        """
        Writes the rest of in memory logs. Currently equivalent to append_to_logfile.
        """
        self.append_to_logfile()

    def write_2D_list(self, list_name, statistics):
        """
        Writes provided 2D list into csv file
        :param list_name: List name used in file creation
        :param statistics: 2D list where each row represents one csv file line
        """
        filename = os.getcwd() + list_name + ".csv"
        print(filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, mode='w+', newline='') as list_file:
            list_writer = csv.writer(list_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for item in statistics:
                list_writer.writerow(item)

    def add_to_plot(self, line_name, points):
        """
        Adds provided points into plot in form of line. This method does not save into file
        Call draw_plot to save the chart.
        :param line_name: Line label that will be displayed in legend
        :param points: list of floats
        """
        points = [x * 100 for x in points]
        plt.plot(points, label=line_name)

    def draw_plot(self, plot_name, file_name, num_of_tests):
        """
        Saves added lines into png chart file and prepares plot for new line addition.
        :param plot_name: Chart name that will be displayed in img
        :param file_name: Name of a file to which should this chart be saved
        :param num_of_tests: Number of performed tests
        """
        plt.axis([0, num_of_tests, 0, 100])
        plt.title(plot_name)
        plt.xlabel("Číslo testu")
        plt.ylabel("Přesnost (%)")
        plt.legend()
        path = os.getcwd()+file_name+".png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.clf()

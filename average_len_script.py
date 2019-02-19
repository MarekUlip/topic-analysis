import csv
import os
csv.field_size_limit(2**24)
base_path = os.getcwd() + "\\csv_folder\\"
for i in [2,3]:
    for name in ["train","test"]:
        with open("{}{}\\{}.csv".format(base_path,i,name), encoding='utf-8', errors='ignore') as csvfile:
            csv_read = csv.reader(csvfile, delimiter=";")
            row_count = 0
            word_count = 0
            for row in csv_read:
                row_count += 1
                word_count += len(row[1].split())
            print("Avg {}: {}".format(name, word_count/row_count))

    """with open("test.csv", encoding='utf-8', errors='ignore') as csvfile:
        csv_read = csv.reader(csvfile, delimiter=",")
        row_count = 0
        word_count = 0
        for row in csv_read:
            row_count += 1
            word_count += len(row[2].split())
        print("Avg test: {}".format(word_count/row_count))"""
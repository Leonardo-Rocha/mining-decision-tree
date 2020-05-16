#!/usr/bin/python3

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import numpy as np
import csv
import os
import graphviz
import matplotlib.pyplot as plt


directory = 'datasets'
dataset = 'car-evaluation'
training_input = []
training_output = []

def preprocess_dataset(attributes_list):
    filename = dataset.split('-')[0]
    filename += '.data'
    full_path = os.path.join(directory, dataset) 
    full_path = os.path.join(full_path, filename)
    with open(full_path, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            # convert strings to number
            for i in range(0, len(row) - 1, 1):
                dict_mapping = attributes_list[i]
                key = row[i]
                row[i] = dict_mapping[key]
            # slice the row and append
            training_input.append(row[:len(row)-1])
            training_output.append(row[-1])
            

def main():
    # this is used to map attribute strings to integer values
    attributes_list = [
                    # 'buying': 
                    {'low': 3, 'med': 2, 'high': 1, 'vhigh': 0},
                    # 'maint': 
                    {'low': 3, 'med': 2, 'high': 1, 'vhigh': 0},
                    # 'doors': 
                    {'2': 0, '3': 1, '4': 2, '5more': 3},
                    # 'persons': 
                    {'2': 0, '4': 1, 'more': 2},
                    # 'lug_boot': 
                    {'small': 0, 'med': 1, 'big': 2},
                    # 'safety': 
                    {'low': 0, 'med': 1, 'high':2} 
    ]
    preprocess_dataset(attributes_list)
    # print(training_input)
    # print(training_output)
    clf = DecisionTreeClassifier()
    clf.fit(training_input, training_output)
    print(f' depth = {clf.get_depth()}')
    dot_data = tree.export_graphviz(clf, out_file=None) 
    graph = graphviz.Source(dot_data)
    graph.render("test.pdf")


if __name__ == "__main__":
    main()

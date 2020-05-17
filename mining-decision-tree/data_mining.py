#!/usr/bin/python3

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn import tree
import csv
import os
import graphviz
 

directory = 'datasets'
dataset = 'iris'
training_input = []
training_output = []

def preprocess_dataset(class_dict, attributes_list=None):
    filename = dataset.split('-')[0]
    filename += '.data'
    full_path = os.path.join(directory, dataset) 
    full_path = os.path.join(full_path, filename)
    with open(full_path, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            if attributes_list != None:
                # convert strings to numbers
                for i in range(0, len(row) - 1, 1):
                    dict_mapping = attributes_list[i]
                    key = row[i]
                    row[i] = dict_mapping[key]
            # slice the row and append
            training_input.append(row[:len(row)-1])

            key = row[-1]
            training_output.append(class_dict[key])
            

def main():
    # this is used to map attribute strings to integer values
    # car_eval_attributes_list = [
    #                 # 'buying': 
    #                 {'low': 3, 'med': 2, 'high': 1, 'vhigh': 0},
    #                 # 'maint': 
    #                 {'low': 3, 'med': 2, 'high': 1, 'vhigh': 0},
    #                 # 'doors': 
    #                 {'2': 0, '3': 1, '4': 2, '5more': 3},
    #                 # 'persons': 
    #                 {'2': 0, '4': 1, 'more': 2},
    #                 # 'lug_boot': 
    #                 {'small': 0, 'med': 1, 'big': 2},
    #                 # 'safety': 
    #                 {'low': 0, 'med': 1, 'high':2} 
    # ]
    # car_eval_class_dict = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
    # car_eval_feature_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    # car_eval_class_names = ['unacc', 'acc', 'good', 'vgood']
    # preprocess_dataset(car_eval_class_dict, car_eval_attributes_list)
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=7)
    # clf.fit(training_input, training_output)
    iris = load_iris()
    clf.fit(iris.data, iris.target)
    # print(f'depth = {clf.get_depth()}')
    # dot_data = tree.export_graphviz(clf, out_file=None, feature_names=car_eval_feature_names, 
    #                                 class_names=car_eval_class_names, filled=True, rounded=True,  
    #                                 special_characters=True) 
    dot_data = tree.export_graphviz(clf, out_file=None, feature_names=iris.feature_names, 
                                    class_names=iris.target_names, filled=True, rounded=True,  
                                    special_characters=True)                                     
    graph = graphviz.Source(dot_data)
    render_path = os.path.join('generated-trees', dataset)
    graph.render(render_path)


if __name__ == "__main__":
    main()

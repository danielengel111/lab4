from sys import argv
import os
from cross_validation import CrossValidation
from knn import KNN
from metrics import accuracy_score
from normalization import *


def load_data():
    """
    Loads data from path in first argument
    :return: returns data as list of Point
    """
    if len(argv) < 2:
        print('Not enough arguments provided. Please provide the path to the input file')
        exit(1)
    input_path = argv[1]

    if not os.path.exists(input_path):
        print('Input file does not exist')
        exit(1)

    points = []
    with open(input_path, 'r') as f:
        for index, row in enumerate(f.readlines()):
            row = row.strip()
            values = row.split(',')
            points.append(Point(str(index), values[:-1], values[-1]))
    return points


def knn_n_fold(k, n, points, normal_type):
    m = KNN(k,1)
    cv = CrossValidation()
    cv.run_cv(points, n, m, accuracy_score,normal_type, print_fold_score=True)


def l1(points):
    new_points = []
    sum=0
    for point in points:
        for coordinate in point.coordinates:
            sum += coordinate
    for point in points:
        new_coordinates = point.coordinates
        new_coordinates = [(new_coordinates[i]/ sum) for i in range(len(point.coordinates))]
        new_points.append(Point(point.name,new_coordinates,point.label))
    return new_points


def run_knn(points):
    # for k in range(1,31):
    #    m = KNN(k=k)
    #    m.train(points)
    #    print(f'predicted class: {m.predict(points[0])}')
    #    print(f'true class: {points[0].label}')
    #    cv = CrossValidation()
    #    cv.run_cv(points, len(points), m, accuracy_score)
    m = KNN(k=7)
    m.train(points)
    print(f'predicted class: {m.predict(points[0])}')
    print(f'true class: {points[0].label}')
    cv = CrossValidation()
    d=DummyNormalizer()
    cv.run_cv(points, 2, m, accuracy_score,d.transform, print_fold_score=True)
    cv.run_cv(points, 10, m, accuracy_score,d.transform, print_fold_score=True)
    cv.run_cv(points, 20, m, accuracy_score,d.transform, print_fold_score=True)
    knn_n_fold(5, 2, points, l1)


if __name__ == '__main__':
    loaded_points = load_data()
    run_knn(loaded_points)

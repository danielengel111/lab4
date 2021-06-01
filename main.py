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
    """
    run knn with n folds with normalized points
    :param k: k-nn
    :param n: n folds
    :param points: the points to use
    :param normal_type: the normalization of those points
    :return:
    """
    m = KNN(k)
    cv = CrossValidation()
    cv.run_cv(normal_type(points), n, m, accuracy_score,normal_type, print_fold_score=True)





def run_knn(points):
    #for k in range(1,31):
     #   m = KNN(k=k)
      #  m.train(points)
       # print(f'predicted class: {m.predict(points[0])}')
        #print(f'true class: {points[0].label}')
        #cv = CrossValidation()
        #cv.run_cv(points, len(points), m, accuracy_score,d.transform(points))
    print("Question 3:\nK=19")
    m = KNN(k=19)
    m.train(points)
    cv = CrossValidation()
    z=ZNormalizer()
    z.fit(points)
    d = DummyNormalizer()
    sum = SumNormalizer()
    min_max=MinMaxNormalizer()
    min_max.fit(points)
    print("2-fold-cross-validation:")
    cv.run_cv(points, 2, m, accuracy_score,d.transform,print_final_score=False, print_fold_score=True)
    print("10-fold-cross-validation:")
    cv.run_cv(points, 10, m, accuracy_score,d.transform,print_final_score=False, print_fold_score=True)
    print("20-fold-cross-validation:")
    cv.run_cv(points, 20, m, accuracy_score,d.transform,print_final_score=False, print_fold_score=True)
    print("Question 4:\nK=5")
    knn_n_fold(5, 2, points, d.transform)
    knn_n_fold(5, 2, points, sum.l1)
    knn_n_fold(5, 2, points, min_max.transform)
    knn_n_fold(5, 2, points, z.transform)
    print("K=7")
    knn_n_fold(7, 2, points, d.transform)
    knn_n_fold(7, 2, points, sum.l1)
    knn_n_fold(7, 2, points, min_max.transform)
    knn_n_fold(7, 2, points, z.transform)


if __name__ == '__main__':
    loaded_points = load_data()
    run_knn(loaded_points)

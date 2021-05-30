from point import Point
from numpy import mean, var


class DummyNormalizer:
    def fit(self, points):
        pass

    def transform(self, points):
        return points


class SumNormalizer:

    def l1(self, points):
        new_points = []
        sum = []
        for point in points:
            for i in range(len(point.coordinates)):
                if (i < len(sum)):
                    sum[i] += abs(point.coordinates[i])
                else:
                    sum.append(abs(point.coordinates[i]))
        for point in points:
            new_coordinates = point.coordinates
            new_coordinates = [(new_coordinates[i]/ sum[i]) for i in range(len(point.coordinates))]
            new_points.append(Point(point.name, new_coordinates, point.label))
        return new_points

class ZNormalizer:
    def __init__(self):
        self.mean_variance_list = []

    def fit(self, points):
        all_coordinates = [p.coordinates for p in points]
        self.mean_variance_list = []
        for i in range(len(all_coordinates[0])):
            values = [x[i] for x in all_coordinates]
            self.mean_variance_list.append([mean(values), var(values, ddof=1)**0.5])

    def transform(self, points):
        new = []
        for p in points:
            new_coordinates = p.coordinates
            new_coordinates = [(new_coordinates[i] - self.mean_variance_list[i][0]) / self.mean_variance_list[i][1]
                               for i in range(len(p.coordinates))]
            new.append(Point(p.name, new_coordinates, p.label))
        return new

class MinMaxNormalizer:

    def __init__(self):
        self.max_coordinate = []
        self.min_coordinate = []

    def fit (self,points):
        all_coordinates = [p.coordinates for p in points]
        for i in range(len(all_coordinates[0])):
            if (len(self.min_coordinate)<=i):
                self.min_coordinate.append(all_coordinates[0][i])
                self.max_coordinate.append(all_coordinates[0][i])
                for x in all_coordinates:
                    if(self.max_coordinate[i]<x[i]):
                        self.max_coordinate[i]=x[i]
                    if(self.min_coordinate[i]>x[i]):
                        self.min_coordinate[i]=x[i]

    def transform(self,points):
        new_points = []
        for p in points:
            new_coordinates=p.coordinates
            new_coordinates = [(new_coordinates[i] - self.min_coordinate[i]) / (self.max_coordinate[i]-self.min_coordinate[i])
                                for i in range(len(p.coordinates))]
            new_points.append(Point(p.name, new_coordinates, p.label))
        return new_points



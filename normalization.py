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
        sum=0
        for point in points:
            for coordinate in point.coordinates:
                sum += abs(coordinate)
        for point in points:
            new_coordinates = point.coordinates
            new_coordinates = [(new_coordinates[i]/ sum) for i in range(len(point.coordinates))]
            new_points.append(Point(point.name,new_coordinates,point.label))
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
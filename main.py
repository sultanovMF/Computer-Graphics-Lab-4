import sys
import numpy as np
from numpy import sqrt, cos, pi, sin, tan
import pygame
from pygame.locals import *
from numpy.linalg import norm
from pygame.version import ver
from scipy.spatial import distance

# Константы
width = 800
height = 800


def translate(radius_vector, Tx, Ty, Tz):
    transformation_matrix = np.array([[1, 0, 0, Tx],
                                      [0, 1, 0, Ty],
                                      [0, 0, 1, Tz],
                                      [0, 0, 0, 1]])
    return np.matmul(transformation_matrix, radius_vector)


def translate_figure(points_list, Tx, Ty, Tz):
    result = []
    for radius_vector in points_list:
        result.append(translate(radius_vector, Tx, Ty, Tz))
    return result


def rotate_z(radius_vector, angle):
    transformation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0, 0],
                                      [np.sin(angle), np.cos(angle), 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]])

    return np.matmul(transformation_matrix, radius_vector)


def rotate_z_figure(points_list, angle):
    result = []
    for radius_vector in points_list:
        result.append(rotate_z(radius_vector, angle))
    return result


def isometric_transformation(radius_vector, camera_deep):
    transformation_matrix = np.array([[sqrt(3) / sqrt(6), 0, - sqrt(3) / sqrt(6), 0],
                                    [1/sqrt(6), 2 / sqrt(6), 1 / sqrt(6), 0],
                                    [sqrt(2) / sqrt(6), - sqrt(2) / sqrt(6), sqrt(2) / sqrt(6), 0],
                                    [0, 0, 0, 1]], dtype="float64")
    return np.matmul(transformation_matrix, radius_vector)


def change_coordinate_system(radius_vector, Tx, Ty):
    transformation_matrix = np.array([[1, 0, 0, Tx],
                                      [0, -1, 0, Ty],
                                      [0, 0, 0, 0],
                                      [0, 0, 0, 0]], dtype=float)
    return np.matmul(transformation_matrix, radius_vector.T)


def orthogonal_projection(radius_vector):
    return np.array([radius_vector[0], radius_vector[1]]).astype(int)


def scale(radius_vector, scale_coefficient):
    transformation_matrix = np.array([[scale_coefficient, 0, 0, 0],
                                      [0, scale_coefficient, 0, 0],
                                      [0, 0, scale_coefficient, 0],
                                      [0, 0, 0, 1]], dtype=float)
    return np.matmul(transformation_matrix, radius_vector)


def get_cube_edges(list_of_cube_coordinates, list_of_cube_point_projection, edge_length):
    list_of_edges = []
    for i in range(0, len(list_of_cube_coordinates) - 1):
        for j in range(i + 1, len(list_of_cube_coordinates)):
            first = list_of_cube_coordinates[i]
            second = list_of_cube_coordinates[j]
            if sqrt(np.dot(first - second, first - second)) - edge_length < 0.01:
                list_of_edges.append(
                    np.array([list_of_cube_point_projection[i], list_of_cube_point_projection[j]]))
    return list_of_edges

def perspective_transformation(radius_vector, d):
    radius_vector = translate(radius_vector, 0, 0, -d)
    transformation_matrix = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 1.0 / d, 0]], dtype=float)
    z = radius_vector[2]
    
    
    result = np.matmul(transformation_matrix, radius_vector.T)
    radius_vector = translate(result, 0, 0, d)
    result /= (z/ d)

   
    return result

class Figure:
    def __init__(self, vertex_points, center_X, center_Y, center_Z):
        self.vertex_points = vertex_points

        self.center_X = center_X
        self.center_Y = center_Y
        self.center_Z = center_Z

        vertex_points = translate_figure(vertex_points, center_X, center_Y, center_Z)

        self.angle_X = 0
        self.angle_Y = 0
        self.angle_Z = 0

    def rotate_Z(self, angle):
        self.vertex_points = rotate_z_figure(self.vertex_points, angle)

    def scale(self, scale_coefficient):
        result = []
        for radius_vector in self.vertex_points:
            result.append(scale(radius_vector, scale_coefficient))
        self.vertex_points = np.array(result)

    def get_isometric_transformation(self, camera_distance):
        list_point_projection = []
        for point in self.vertex_points:
            point = isometric_transformation(point,  camera_distance)
            point = change_coordinate_system(point, width / 2.0, height / 2.0)
            point = orthogonal_projection(point)
            list_point_projection.append(np.array(point))

        return np.array(list_point_projection)

    def translate(self, Tx, Ty, Tz):
        self.vertex_points = translate_figure(self.vertex_points, Tx, Ty, Tz)

    def get_perspective_transformation(self, d):
        list_point_projection = []
        for point in self.vertex_points:
            point = perspective_transformation(point,  d)

            point = change_coordinate_system(point, width / 2.0, height / 2.0)
            point = orthogonal_projection(point)
            list_point_projection.append(np.array(point))

        return np.array(list_point_projection)

class Cube(Figure):
    def __init__(self, length, center_X, center_Y, center_Z):

        vertex_points = np.array([np.array([0.5, 0.5, 0.5, 1]),
                                  np.array([0.5, 0.5, -0.5, 1]),
                                  np.array([0.5, -0.5, 0.5, 1]),
                                  np.array([0.5, -0.5, -0.5, 1]),
                                  np.array([-0.5, 0.5, 0.5, 1]),
                                  np.array([-0.5, 0.5, -0.5, 1]),
                                  np.array([-0.5, -0.5, 0.5, 1]),
                                  np.array([-0.5, -0.5, -0.5, 1])], dtype="float64")
        
        self.length = length
        Figure.__init__(self, vertex_points, center_X, center_Y, center_Z)

        self.scale(length)

    def scale(self, scale_coefficient):
        Figure.scale(self, scale_coefficient)
        self.length = scale_coefficient

    def draw_isometric(self, surface, color, camera_distance):
        list_of_cube_point_projection = self.get_isometric_transformation(camera_distance)

        for point in list_of_cube_point_projection:
            pygame.draw.circle(surface, color, point, 3)

        for edge in get_cube_edges(self.vertex_points, list_of_cube_point_projection, self.length):
            pygame.draw.line(surface, color, edge[0], edge[1])

    def draw_perspective(self, surface, color, d):
        
        list_of_cube_point_projection = self.get_perspective_transformation(d)

        for point in list_of_cube_point_projection:
            pygame.draw.circle(surface, color, point, 3)

        for edge in get_cube_edges(self.vertex_points, list_of_cube_point_projection, self.length):
            pygame.draw.line(surface, color, edge[0], edge[1])

class VerticalPrism(Figure):
    def __init__(self, length, radius, center_X, center_Y, center_Z):
        vertex_points = [
                        # правая часть 
                        np.array([-1 * radius, length / 2.0, 0, 1]),
                        np.array([-1 / sqrt(2) * radius, length / 2.0, 1 / sqrt(2) * radius, 1]),
                        np.array([1 / sqrt(2) * radius, length / 2.0, 1 / sqrt(2) * radius, 1]),
                        np.array([1 * radius, length / 2.0, 0, 1]),
                        np.array([1 / sqrt(2) * radius, length / 2.0, -1 / sqrt(2) * radius, 1]),
                        np.array([-1 / sqrt(2) * radius, length / 2.0, -1 / sqrt(2) * radius, 1]),
                        # левая
                        np.array([-1 * radius, -length / 2.0, 0, 1]),
                        np.array([-1 / sqrt(2) * radius, -length / 2.0, 1 / sqrt(2) * radius, 1]),
                        np.array([1 / sqrt(2) * radius, -length / 2.0, 1 / sqrt(2) * radius, 1]),
                        np.array([1 * radius, -length / 2.0, 0, 1]),
                        np.array([1 / sqrt(2) * radius, -length / 2.0, -1 / sqrt(2) * radius, 1]),
                        np.array([-1 / sqrt(2) * radius, -length / 2.0, -1 / sqrt(2) * radius, 1])]

        Figure.__init__(self, vertex_points, center_X, center_Y, center_Z)
        self.length = length
        self.radius = radius


    def scale(self, scale_coefficient):
        Figure.scale(self, scale_coefficient)
        self.length *= scale_coefficient

    def draw_isometric(self, surface, color, camera_distance):
        list_of_cube_point_projection = self.get_isometric_transformation(camera_distance)

        for point in list_of_cube_point_projection:
            pygame.draw.circle(surface, color, point, 3)

        for i in range(0, 5):
            pygame.draw.line(surface, color, list_of_cube_point_projection[i], list_of_cube_point_projection[i + 1])
        pygame.draw.line(surface, color, list_of_cube_point_projection[0], list_of_cube_point_projection[5])

        for i in range(6, 11):
            pygame.draw.line(surface, color, list_of_cube_point_projection[i], list_of_cube_point_projection[i + 1])
        pygame.draw.line(surface, color, list_of_cube_point_projection[6], list_of_cube_point_projection[11])

        for i in range(0, 6):
            pygame.draw.line(surface, color, list_of_cube_point_projection[i], list_of_cube_point_projection[i + 6])

    def draw_perspective(self, surface, color, d):
        list_of_cube_point_projection = self.get_perspective_transformation(d)

        print(list_of_cube_point_projection)
        for point in list_of_cube_point_projection:
            pygame.draw.circle(surface, color, point, 3)

        for i in range(0, 5):
            pygame.draw.line(surface, color, list_of_cube_point_projection[i], list_of_cube_point_projection[i + 1])
        pygame.draw.line(surface, color, list_of_cube_point_projection[0], list_of_cube_point_projection[5])

        for i in range(6, 11):
            pygame.draw.line(surface, color, list_of_cube_point_projection[i], list_of_cube_point_projection[i + 1])
        pygame.draw.line(surface, color, list_of_cube_point_projection[6], list_of_cube_point_projection[11])

        for i in range(0, 6):
            pygame.draw.line(surface, color, list_of_cube_point_projection[i], list_of_cube_point_projection[i + 6])

class HorizontalPrism(Figure):
    def __init__(self, length, radius, center_X, center_Y, center_Z):
        vertex_points = [
                        #верхняя часть 
                        np.array([-1 * radius, 0, length / 2.0, 1]),
                        np.array([-1 / sqrt(2) * radius, 1 / sqrt(2) * radius, length / 2.0, 1]),
                        np.array([1 / sqrt(2) * radius, 1 / sqrt(2) * radius, length / 2.0, 1]),
                        np.array([1 * radius, 0, length / 2.0, 1]),
                        np.array([1 / sqrt(2) * radius, -1 / sqrt(2) * radius, length / 2.0, 1]),
                        np.array([-1 / sqrt(2) * radius, -1 / sqrt(2) * radius, length / 2.0, 1]),
                        # нижняя часть
                        np.array([-1 * radius, 0, -length / 2.0, 1]),
                        np.array([-1 / sqrt(2) * radius, 1 / sqrt(2) * radius, -length / 2.0, 1]),
                        np.array([1 / sqrt(2) * radius, 1 / sqrt(2) * radius, -length / 2.0, 1]),
                        np.array([1 * radius, 0, -length / 2.0, 1]),
                        np.array([1 / sqrt(2) * radius, -1 / sqrt(2) * radius, -length / 2.0, 1]),
                        np.array([-1 / sqrt(2) * radius, -1 / sqrt(2) * radius, -length / 2.0, 1])]

        Figure.__init__(self, vertex_points, center_X, center_Y, center_Z)
        self.length = length
        self.radius = radius

    def scale(self, scale_coefficient):
        Figure.scale(self, scale_coefficient)
        self.length *= scale_coefficient

    def draw_isometric(self, surface, color, camera_distance):
        list_of_cube_point_projection = self.get_isometric_transformation(camera_distance)

        for point in list_of_cube_point_projection:
            pygame.draw.circle(surface, color, point, 3)

        for i in range(0, 5):
            pygame.draw.line(surface, color, list_of_cube_point_projection[i], list_of_cube_point_projection[i + 1])
        pygame.draw.line(surface, color, list_of_cube_point_projection[0], list_of_cube_point_projection[5])

        for i in range(6, 11):
            pygame.draw.line(surface, color, list_of_cube_point_projection[i], list_of_cube_point_projection[i + 1])
        pygame.draw.line(surface, color, list_of_cube_point_projection[6], list_of_cube_point_projection[11])

        for i in range(0, 6):
            pygame.draw.line(surface, color, list_of_cube_point_projection[i], list_of_cube_point_projection[i + 6])

    def draw_perspective(self, surface, color, d):
        list_of_cube_point_projection = self.get_perspective_transformation(d)

        print(list_of_cube_point_projection)
        for point in list_of_cube_point_projection:
            pygame.draw.circle(surface, color, point, 3)

        for i in range(0, 5):
            pygame.draw.line(surface, color, list_of_cube_point_projection[i], list_of_cube_point_projection[i + 1])
        pygame.draw.line(surface, color, list_of_cube_point_projection[0], list_of_cube_point_projection[5])

        for i in range(6, 11):
            pygame.draw.line(surface, color, list_of_cube_point_projection[i], list_of_cube_point_projection[i + 1])
        pygame.draw.line(surface, color, list_of_cube_point_projection[6], list_of_cube_point_projection[11])

        for i in range(0, 6):
            pygame.draw.line(surface, color, list_of_cube_point_projection[i], list_of_cube_point_projection[i + 6])



if __name__ == "__main__":
    pygame.init()

    cube = Cube(200, 0, 0, 0)

    vertical_prism = VerticalPrism(200, 50, 0, 0, 0)
    horizontal_prism = HorizontalPrism(200, 50, 0, 0, 0)
    surface = pygame.display.set_mode((width, height))

    angle = 0
    da = np.pi / 36

    dx = 10
    Tx = 0
    while True:

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                pass

        surface.fill((0, 0, 0))

        # Рисование системы координат
        pygame.draw.line(surface, (255, 255, 255),
                         (width // 2, 0), (width // 2, height))
        pygame.draw.line(surface, (255, 255, 255),
                         (0, height // 2), (width, height // 2))

        pygame.time.wait(100)
        
        if (Tx < -40 or Tx > 40):
            dx *= -1
        
        Tx += dx
        #vertical_prism.translate(Tx, 0, 0)
        #vertical_prism.draw_isometric(surface, (255, 0, 0), 1)
        # vertical_prism.draw_perspective(surface, (255, 0, 0), 400)

        horizontal_prism.translate(Tx, 0, 0)
        # horizontal_prism.draw_isometric(surface, (255, 0, 0), 1)
        horizontal_prism.draw_perspective(surface, (255, 0, 0), 400)

        # cube.translate(0, 0, Tx)
        # # cube.draw_isometric(surface, (0, 255, 0), 1)
        # cube.draw_perspective(surface, (255, 0, 0), 400)
    
    

        angle += da
        pygame.display.update()

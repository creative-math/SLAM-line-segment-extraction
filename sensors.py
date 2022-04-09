import pygame
import numpy as np


def uncertanty_add(distance, angle, sigma):
    mean = np.array([distance, angle])
    covariance = np.diag(sigma ** 2)
    distance, angle = np.random.multivariate_normal(mean, covariance)
    distance = max(distance, 0)
    angle = max(angle, 0)
    return [distance, angle]


class LaserSensor:

    def __init__(self, Range, nAngles, map, uncertanty):
        self.Range = Range
        self.nAngles = nAngles
        self.map = map
        self.speed = 4  # rounds per second
        self.sigma = np.array([uncertanty[0], uncertanty[1]])
        self.position = np.zeros(2)
        self.W, self.H = pygame.display.get_surface().get_size()
        self.sensedObstacles = []

        numpyMap = np.zeros((self.W, self.H), dtype=np.bool_)
        it = np.nditer(numpyMap, flags=['multi_index'], op_flags=['writeonly'])
        for x in it:
            color = self.map.get_at(it.multi_index)
            if [color[0], color[1], color[2]] <= [10, 10, 10]:
                x[...] = True
        self.solid = np.pad(numpyMap, 1, 'constant', constant_values=False)

        radius = np.linspace(0, Range, int(Range / 2), False)
        self.angles = np.linspace(0, 2 * np.pi, nAngles, False)
        anglesXY = np.append(np.expand_dims(np.cos(self.angles), 1), np.expand_dims(np.sin(self.angles), 1), 1)
        self.pixels = np.array(np.round(np.expand_dims(np.expand_dims(radius, 1), 0) * np.expand_dims(anglesXY, 1)),
                               dtype=int)
        self.all_pixel_distances = np.linalg.norm(self.pixels, axis=2)

    # def distance(self, obstaclePosition):
    #     return np.linalg.norm(np.array(obstaclePosition) - np.array(self.position))
    #     # px = (obstaclePosition[0] - self.position[0]) ** 2
    #     # py = (obstaclePosition[1] - self.position[1]) ** 2
    #     # return np.sqrt(px + py)

    def sense_obstacles(self):
        data = []
        mask = self.solid[tuple(np.clip(self.pixels + (self.position + np.ones(2, dtype=int)),
                                        np.zeros(2, dtype=int), np.array([self.W + 1, self.H + 1])).T)].T
        distances = np.min(np.where(mask, self.all_pixel_distances, self.Range + 1), 1)

        angles = self.angles[distances <= self.Range]
        distances = distances[distances <= self.Range]
        # TODO: store data in one numpy array (only distances and angles, plus position in first entry, or separately)
        for i in range(0, len(angles)):
            data.append([distances[i], angles[i], self.position])

        if len(data) > 0:
            return data
        else:
            return False

        # x1, y1 = self.position[0], self.position[1]
        # for angle in np.linspace(0, 2 * np.pi, self.nAngles, False):  # originally divided into 60 angles
        #     x2, y2 = (x1 + self.Range * np.cos(angle), y1 + self.Range * np.sin(angle))
        #     for i in range(0, 100):
        #         u = i / 100
        #         x = int(x2 * u + x1 * (1 - u))  # x = x1 + u * (x2 - x1)
        #         y = int(y2 * u + y1 * (1 - u))
        #         if 0 < x < self.W and 0 < y < self.H:
        #             color = self.map.get_at((x, y))
        #             # if (color[0], color[1], color[2] == (0, 0, 0)):
        #             # if color[0] == 0 and color[1] == 0 and color[2] == 0:
        #             if [color[0], color[1], color[2]] <= [10, 10, 10]:
        #                 distance = self.distance((x, y))
        #
        #                 # output = uncertanty_add(distance, angle, self.sigma)
        #                 # output.append(self.position)
        #                 output = [distance, angle, self.position, (x, y)]  # uncomment to skip uncertanty
        #
        #                 # store the measurements
        #                 data.append(output)
        #                 break

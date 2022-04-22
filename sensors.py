import pygame
import numpy as np

add_uncertainty = True  # set False to sense exact data


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

    def sense_obstacles(self):
        mask = self.solid[tuple(np.clip(self.pixels + (self.position + np.ones(2, dtype=int)),
                                        np.zeros(2, dtype=int), np.array([self.W + 1, self.H + 1])).T)].T
        distances = np.min(np.where(mask, self.all_pixel_distances, self.Range + 1), 1)
        data = [distances[distances <= self.Range], self.angles[distances <= self.Range], self.position]
        if add_uncertainty and data[0].size > 0:
            data[0] = np.clip(data[0] + self.sigma[0] * np.random.randn(data[0].size), a_min=0, a_max=None)
            data[1] = np.clip(data[1] + self.sigma[1] * np.random.randn(data[1].size), a_min=0, a_max=2 * np.pi)
        return data if data[0].size > 0 else False

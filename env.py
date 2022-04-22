import numpy as np
import pygame


class buildEnvironment:
    def __init__(self, MapDimensions, MapName):
        pygame.init()
        self.pointCloud = np.zeros((0, 2), dtype=int)
        self.externalMap = pygame.image.load(MapName)
        self.maph, self.mapw = MapDimensions
        self.MapWindowName = 'RRT path planning'
        pygame.display.set_caption(self.MapWindowName)
        self.map = pygame.display.set_mode((self.mapw, self.maph))  # Create a canvas on which to display everything
        self.map.blit(self.externalMap, (0, 0))  # Blit the image onto the canvas
        self.black = (0, 0, 0)
        self.grey = (70, 70, 70)
        self.Blue = (0, 0, 255)
        self.Green = (0, 255, 0)
        self.Red = (255, 0, 0)
        self.white = (255, 255, 255)
        self.infomap = self.map.copy()

    @staticmethod
    def AD2pos(distances, angles, robotPosition):
        # bug fix: no minus before distance on the y coordinate
        return robotPosition + np.expand_dims(distances, 1) * np.append(np.expand_dims(np.cos(angles), 1),
                                                                        np.expand_dims(np.sin(angles), 1), axis=1)

    def dataStorage(self, data):
        points = []
        if data:  # convert angle, distance and position to pixel
            points = np.array(self.AD2pos(data[0], data[1], data[2]), dtype=int)
            self.pointCloud = np.unique(np.append(self.pointCloud, points, axis=0), axis=0)
        return points

    def show_sensorData(self, data):
        for point in data:
            self.infomap.set_at(point, (255, 0, 0))

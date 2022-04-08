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
        self.map = pygame.display.set_mode((self.mapw, self.maph))
        self.map.blit(self.externalMap, (0, 0))
        self.black = (0, 0, 0)
        self.grey = (70, 70, 70)
        self.Blue = (0, 0, 255)
        self.Green = (0, 255, 0)
        self.Red = (255, 0, 0)
        self.white = (255, 255, 255)

    # TODO: vectorize AD2pos
    def AD2pos(self, distance, angle, robotPosition):
        x = distance * np.cos(angle) + robotPosition[0]
        y = distance * np.sin(angle) + robotPosition[1]  # bug fix: minus before distance
        return (int(x), int(y))

    def dataStorage(self, data):
        if data:  # bug fix from Louis Diedericks (YTB Comments)
            points = []
            for element in data:
                point = np.array(self.AD2pos(element[0], element[1], element[2]), dtype=int)  # convert angle, distance and position to pixel
                points.append(point)
            self.pointCloud = np.unique(np.append(self.pointCloud, np.array(points), axis=0), axis=0)
            # if point not in self.pointCloud:
            #     self.pointCloud.append(point)

    # TODO: draw only new points and save the old ones to shorten the for loop
    def show_sensorData(self):
        self.infomap = self.map.copy()
        for point in self.pointCloud:  # is it possible to input an array into set_at(...)?
            self.infomap.set_at((int(point[0]), int(point[1])), (255, 0, 0))

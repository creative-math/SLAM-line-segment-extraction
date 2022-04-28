import pygame
import random
import time

import env
import sensors
import features

detect_lines = True  # set False to show only the sensed laser points
detect_landmarks = True  # set False to stop data association


def random_color():
    levels = range(32, 256, 32)
    return tuple(random.choice(levels) for _ in range(3))


if __name__ == '__main__':
    # initiate the internal Map, the lidar some other relevant variables
    FeatureMAP = features.featuresDetection()
    environment = env.buildEnvironment((600, 1200), 'src/map1.png')
    originalMap = environment.map.copy()
    laser = sensors.LaserSensor(200, 200, originalMap, uncertanty=(0.5, 0.01))
    environment.map.fill((0, 0, 0))
    environment.infomap = environment.map.copy()
    originalMap = environment.map.copy()
    running = True
    FEATURE_DETECTION = True
    BREAK_POINT_IND = 0

    num_events = 0
    sum_time = 0

    while running:
        FEATURE_DETECTION = True
        BREAK_POINT_IND = 0
        ENDPOINTS = [0, 0]
        sensorON = False
        PREDICTED_POINTS_TODRAW = []
        # check that the game is still running with the mouse on the environment
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("\nNumber of Events: %s - total time: %s sec - time per event: %s sec" %
                      (num_events, round(sum_time, 3), round(sum_time / num_events, 3)))
                print("")
                running = False
            if pygame.mouse.get_focused():
                sensorON = True
            elif not pygame.mouse.get_focused():
                sensorON = False
        # start a new scanning round
        if sensorON:
            # update the current mouse / robot position
            position = pygame.mouse.get_pos()
            laser.position = position

            # erase the last located features
            environment.infomap = originalMap.copy()
            # visualize the current scanning radius
            pygame.draw.circle(environment.infomap, (255, 0, 0), laser.position, laser.Range, 2)
            pygame.draw.line(environment.infomap, (255, 0, 0), laser.position,
                             (laser.position[0] + laser.Range, laser.position[1]))
            # create an array that contains for every angle the distance to the first black pixel and the robot position
            sensor_data = laser.sense_obstacles()
            # convert the measured distances and angles to pixels, depending on the current robot position
            FeatureMAP.laser_points_set(sensor_data)

            for point in FeatureMAP.LASERPOINTS:
                environment.infomap.set_at((int(point[0]), int(point[1])), (255, 0, 0))

            if not sensor_data:
                environment.map.blit(environment.infomap, (0, 0))
                pygame.display.update()
                continue

            start_time = time.time()

            # while there are more unworked points left than a segment needs to have at least
            while BREAK_POINT_IND < (FeatureMAP.NP - FeatureMAP.PMIN) and detect_lines:

                # try to detect a seed segment, if this fails, no further seed segments can be found
                seedSeg = FeatureMAP.seed_segment_detection(laser.position, BREAK_POINT_IND)
                if seedSeg == False:
                    break
                else:
                    # assign a separate variable to each output of the segment detection
                    seedSegment = seedSeg[0]
                    PREDICTED_POINTS_TODRAW = seedSeg[1]
                    INDICES = seedSeg[2]
                    # TODO: clip the index to allow segment growing between array start and end
                    results = FeatureMAP.seed_segment_growing(INDICES, BREAK_POINT_IND)
                    # if the region growing created a line segment, that is long enough and contains enough points
                    if results == False:
                        BREAK_POINT_IND = INDICES[1]
                        continue
                    else:
                        # assign a separate variable to each output of the region growing
                        line_eq = results[1]
                        m, c = results[5]
                        line_seg = results[0]
                        OUTERMOST = results[2]
                        BREAK_POINT_IND = results[3]

                        # calculate the start and end pixel of the scanned line, so it can be drawn on the infomap
                        ENDPOINTS[0] = FeatureMAP.projection_point2line(OUTERMOST[0], m, c)
                        ENDPOINTS[1] = FeatureMAP.projection_point2line(OUTERMOST[1], m, c)

                        if detect_landmarks:
                            FeatureMAP.FEATURES.append([[m, c], ENDPOINTS])
                            pygame.draw.line(environment.infomap, (0, 255, 0), ENDPOINTS[0], ENDPOINTS[1], 1)
                            environment.dataStorage(sensor_data)

                            FeatureMAP.FEATURES = FeatureMAP.lineFeats2point()
                            features.landmark_association(FeatureMAP.FEATURES)

                        else:
                            COLOR = random_color()
                            for point in line_seg:
                                environment.infomap.set_at((int(point[0]), int(point[1])), (0, 255, 0))
                                pygame.draw.circle(environment.infomap, COLOR, (int(point[0]), int(point[1])), 2, 0)
                            pygame.draw.line(environment.infomap, (255, 0, 0), ENDPOINTS[0], ENDPOINTS[1], 2)

                            environment.dataStorage(sensor_data)
            if detect_lines:
                sum_time += time.time() - start_time
            num_events += 1
        if detect_landmarks:
            for landmark in features.Landmarks:
                pygame.draw.line(environment.infomap, (0, 0, 255), landmark[1][0], landmark[1][1], 2)

        environment.map.blit(environment.infomap, (0, 0))
        pygame.display.update()

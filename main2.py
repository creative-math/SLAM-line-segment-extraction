import pygame
import random
import time
import numpy as np

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
    FeatureMAP = features.FeaturesDetection()
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

            # collect Lidar data
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

            # start detecting seed segments and grow their region
            # ---------------------------------------------------------------------------------------------
            while BREAK_POINT_IND < (FeatureMAP.NP - FeatureMAP.PMIN) and detect_lines:
                # detect the next segment
                seedSeg = FeatureMAP.seed_segment_detection(laser.position, BREAK_POINT_IND)
                if not seedSeg:
                    break
                else:
                    seedSegment = seedSeg[0]
                    INDICES = seedSeg[1]

                    # grow the region of the segment, it has to contain enough points and has to be long enough
                    results = FeatureMAP.seed_segment_growing(INDICES, BREAK_POINT_IND)
                    if not results:
                        BREAK_POINT_IND = INDICES[1]
                        continue
                    else:
                        line_eq = results[0]
                        PB = results[1]
                        PF = results[2]
                        BREAK_POINT_IND = PF

                        # calculate the start and end pixel of the scanned line, so it can be drawn on the infomap
                        ENDPOINTS = FeatureMAP.projection_point2line(
                            line_eq, FeatureMAP.LASERPOINTS[[PB, PF - 1]])

                        if detect_landmarks:  # save the current line segment
                            FeatureMAP.FEATURES.append([line_eq, ENDPOINTS])
                        else:  # draw the current line segment
                            COLOR = random_color()
                            for point in FeatureMAP.LASERPOINTS[PB:PF]:
                                pygame.draw.circle(environment.infomap, COLOR, (int(point[0]), int(point[1])), 2, 0)
                            pygame.draw.line(environment.infomap, (255, 0, 0), ENDPOINTS[0], ENDPOINTS[1], 2)

            if detect_landmarks and detect_lines:  # start data association
                new_rep = []  # new feature representation holds the line params, start & end point and a projection
                for feature in FeatureMAP.FEATURES:
                    # draw the detected line in green
                    pygame.draw.line(environment.infomap, (0, 255, 0), feature[1][0], feature[1][1], 1)
                    new_rep.append([feature[0], feature[1], FeatureMAP.projection_point2line(feature[0], np.zeros(2))])
                FeatureMAP.landmark_association(new_rep)
                FeatureMAP.FEATURES = []

            if detect_lines:
                sum_time += time.time() - start_time
            num_events += 1

        if detect_landmarks:  # draw all stored landmarks
            for landmark in FeatureMAP.LANDMARKS:
                pygame.draw.line(environment.infomap, (0, 0, 255), landmark[1][0], landmark[1][1], 2)

        environment.map.blit(environment.infomap, (0, 0))
        pygame.display.update()

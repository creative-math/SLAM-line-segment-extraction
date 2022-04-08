import pygame
import numpy as np
import time

import env
import sensors

environment = env.buildEnvironment((600, 1200), 'map2.png')
environment.originalMap = environment.map.copy()
laser = sensors.LaserSensor(200, 200, environment.originalMap, uncertanty=(0.5, 0.01))
environment.map.fill((0, 0, 0))
environment.infomap = environment.map.copy()
# originalMap = environment.map.copy()  # added to provide the circle

running = True
num_events = 0
sum_time = 0

while running:
    # environment.infomap = originalMap.copy()  # added to provide the circle
    sensorON = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print("\nNumber of Events: %s - total time: %s sec - time per event: %s sec" %
                  (num_events, round(sum_time, 3), round(sum_time / num_events, 3)))
            running = False
        if pygame.mouse.get_focused():
            sensorON = True
        elif not pygame.mouse.get_focused():
            sensorON = False
    if sensorON:
        start_time = time.time()
        position = pygame.mouse.get_pos()
        # laser.position[0] = position[0]
        # laser.position[1] = position[1]
        laser.position = np.array(position)
        sensor_data = laser.sense_obstacles()  # 1D-array of max len 60, that contains for every angle the distance to the first black pixel and the original position
        print("\rNumber of stored points: %s" % len(environment.pointCloud), end='')
        environment.dataStorage(sensor_data)
        environment.show_sensorData()
        # pygame.draw.circle(environment.infomap, (255, 0, 0), laser.position, laser.Range, 2)  # draw the scan radius
        sum_time += time.time() - start_time
        num_events += 1
    environment.map.blit(environment.infomap, (0, 0))
    pygame.display.update()

# pygame.quit()  # bug fix von Louis

import numpy as np

# landmarks
Landmarks = []


class featuresDetection:
    def __init__(self):
        self.LASERPOINTS = np.zeros(0)
        # variables
        self.EPSILON = 10
        self.DELTA = 501
        self.SNUM = 6
        self.PMIN = 20  # index of the scanned point, that is currently active
        self.GMAX = 20
        self.SEED_SEGMENTS = []
        self.LINE_SEGMENTS = []
        self.LINE_PARAMS = None
        self.NP = len(self.LASERPOINTS) - 1  # number of scanned points
        self.LMIN = 20  # minimum length of a line segment (originally 20)
        self.LR = 0  # real length of a line segment
        self.PR = 0  # the number of laser points contained in the line segment
        self.FEATURES = []

    # euclidian distance from point1 to point2
    @staticmethod
    def dist_point2point(point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    # distance point to line written in the general form
    @staticmethod
    def dist_point2line(params, point):
        a, b, c = params
        vector = np.array([a, b])
        return np.abs(np.dot(point, vector) + c) / np.linalg.norm(vector)

    # extract two points from a line equation under the slope intercepts from
    @staticmethod
    def line_2points(m, b):
        x = 5
        y = m * x + b
        x2 = 2000
        y2 = m * x2 + b
        return [(x, y), (x2, y2)]

    # general form to slope-intercept
    @staticmethod
    def lineForm_G2SI(A, B, C):
        m = -A / B
        B = -C / B
        return m, B

    @staticmethod
    def projection_point2line(point, m, b):
        x, y = point
        if m != 0:
            m2 = -1 / m
            c2 = y - m2 * x
            intersection_x = - (b - c2) / (m - m2)
            intersection_y = m2 * intersection_x + c2
        else:
            return x, b
        return intersection_x, intersection_y

    @staticmethod
    def AD2pos(distances, angles, robotPosition):
        return robotPosition + np.expand_dims(distances, 1) * np.append(np.expand_dims(np.cos(angles), 1),
                                                                        np.expand_dims(np.sin(angles), 1), axis=1)

    @staticmethod
    def intersect2lines(line_params, laserpoint, robotpos):
        a, b, c = line_params
        v_rot = np.array([a, b])  # 90° rotated directional vector of the line
        ba = laserpoint - robotpos  # directional vector of the laser scan
        # two parallel lines intersect in infinity
        return robotpos + ba * (-c - np.dot(robotpos, v_rot)) / \
            np.expand_dims(np.dot(ba, v_rot), len(laserpoint.shape) - 1) \
            if np.all(np.dot(ba, v_rot) != 0) else np.full(laserpoint.shape, np.inf)

    @staticmethod
    def odr_fit(data):  # orthogonal distance regression
        data_mean = np.mean(data, axis=0)
        _, _, V = np.linalg.svd(data - data_mean)  # singular value decomposition
        b = 1e-10 if V[1, 1] == 0 else V[1, 1] * 1e4  # b should not equal zero to prevent division by zero errors
        return -V[1, 0] * -1e4, b, np.dot(data_mean, V[1]) * -1e4  # numbers a, b, c mustn't get too small

    def laser_points_set(self, data):
        self.LASERPOINTS = []
        if not data:
            pass
        else:  # convert distance, angle and position to pixel
            self.LASERPOINTS = np.array(self.AD2pos(data[0], data[1], data[2]), dtype=int)
        self.NP = len(self.LASERPOINTS) - 1

    def seed_segment_detection(self, robot_position, break_point_ind):
        self.NP = max(0, self.NP)
        self.SEED_SEGMENTS = []
        # NP = Number (laser-) Points, PMIN = Min Number of Points a seed segment should have
        for i in range(break_point_ind, (self.NP - self.PMIN)):
            j = i + self.SNUM  # SNUM = Number of points in our seed segment
            params = self.odr_fit(self.LASERPOINTS[i:j])
            predicted_points = self.intersect2lines(params, self.LASERPOINTS[i:j], robot_position)

            # if the fitted line fulfills the epsilon and delta condition
            if np.all(np.linalg.norm(predicted_points - self.LASERPOINTS[i:j], axis=1) <= self.DELTA) \
                    and np.all(self.dist_point2line(params, predicted_points) <= self.EPSILON):
                self.LINE_PARAMS = params
                return [self.LASERPOINTS[i:j], (i, j)]
        return False

    def seed_segment_growing(self, indices, break_point):
        line_eq = self.LINE_PARAMS
        i, j = indices
        # Beginning and Final points in the line segment
        PB, PF = max(break_point, i - 1), min(j + 1, len(self.LASERPOINTS) - 1)

        while self.dist_point2line(line_eq, self.LASERPOINTS[PF]) < self.EPSILON:
            if PF > self.NP - 1:
                break
            elif PB <= PF:
                line_eq = self.odr_fit(self.LASERPOINTS[PB:PF])

                POINT = self.LASERPOINTS[PF]
            else:
                break

            PF = PF + 1
            NEXTPOINT = self.LASERPOINTS[PF]
            if self.dist_point2point(POINT, NEXTPOINT) > self.GMAX:
                break

        PF = PF - 1

        while self.dist_point2line(line_eq, self.LASERPOINTS[PB]) < self.EPSILON:
            if PB < break_point:
                break
            elif PF <= PB:
                line_eq = self.odr_fit(self.LASERPOINTS[PF:PB])
                POINT = self.LASERPOINTS[PB]
            else:
                break

            PB = PB - 1
            NEXTPOINT = self.LASERPOINTS[PB]
            if self.dist_point2point(POINT, NEXTPOINT) > self.GMAX:
                break
        PB = PB + 1

        LR = self.dist_point2point(self.LASERPOINTS[PB], self.LASERPOINTS[PF])
        PR = len(self.LASERPOINTS[PB:PF])

        if (LR >= self.LMIN) and (PR >= self.PMIN):
            self.LINE_PARAMS = line_eq
            m, b = self.lineForm_G2SI(line_eq[0], line_eq[1], line_eq[2])
            self.two_points = self.line_2points(m, b)
            self.LINE_SEGMENTS.append((self.LASERPOINTS[PB + 1], self.LASERPOINTS[PF - 1]))
            return [self.LASERPOINTS[PB:PF], self.two_points,
                    (self.LASERPOINTS[PB + 1], self.LASERPOINTS[PF - 1]), PF, line_eq, (m, b)]
        else:
            return False

    def lineFeats2point(self):
        new_rep = []  # the new representation of tne features

        for feature in self.FEATURES:
            projection = self.projection_point2line((0, 0), feature[0][0], feature[0][1])
            new_rep.append([feature[0], feature[1], projection])

        return new_rep


def landmark_association(landmarks):
    thresh = 10
    for l in landmarks:

        flag = False
        for i, Landmark in enumerate(Landmarks):
            dist = featuresDetection.dist_point2point(l[2], Landmark[2])
            if dist < thresh:
                if not is_overlap(l[1], Landmark[1]):
                    continue
                else:
                    Landmarks.pop(i)
                    Landmarks.insert(i, l)
                    flag = True

                    break
        if not flag:
            Landmarks.append(l)


def is_overlap(seg1, seg2):
    length1 = featuresDetection.dist_point2point(seg1[0], seg1[1])
    length2 = featuresDetection.dist_point2point(seg2[0], seg2[1])
    center1 = ((seg1[0][0] + seg1[1][0]) / 2, (seg1[0][1] + seg1[1][1]) / 2)
    center2 = ((seg2[0][0] + seg2[1][0]) / 2, (seg2[0][1] + seg2[1][1]) / 2)
    dist = featuresDetection.dist_point2point(center1, center2)
    if dist > (length1 + length2) / 2:
        return False
    else:
        return True

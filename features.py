import numpy as np

# landmarks
Landmarks = []


class featuresDetection:
    def __init__(self):
        # variables for storing things
        self.LASERPOINTS = np.zeros(0)
        self.SEED_SEGMENTS = []
        self.LINE_SEGMENTS = []
        self.LINE_PARAMS = None
        self.FEATURES = []

        # variables in Pixel units
        self.EPSILON = 10  # maximum orthogonal distance of a point to its line
        self.DELTA = 50  # maximum non-orthogonal distance of a point to its line
        self.Kappa = 2 * self.EPSILON  # this variable is necessary, because we delete measurements without obstacles
        self.LMIN = 20  # minimum length of a line segment (originally 20)

        # variables for counting points
        self.NP = len(self.LASERPOINTS) - 1  # number of scanned points
        self.SNUM = 6  # number of points to start segment detection with
        self.PMIN = 10  # number of points a line segment has to have at least
        self.STEP = min(self.SNUM, self.PMIN - self.SNUM)  # step size in segment detection

    # distance point to line written in the general form
    @staticmethod
    def dist_point2line(params, point):
        a, b, c = params
        w = np.array([a, b])  # orthogonal vector to the line
        return np.abs(np.dot(point, w) + c) / np.linalg.norm(w)

    @staticmethod
    def projection_point2line(params, point):
        a, b, c = params
        w = np.array([a, b])
        return point - w * (np.dot(point, w) + c) / (np.linalg.norm(w) ** 2)

    @staticmethod  # Angles, Distances to position
    def AD2pos(distances, angles, robotPosition):
        return robotPosition + np.expand_dims(distances, 1) * np.append(np.expand_dims(np.cos(angles), 1),
                                                                        np.expand_dims(np.sin(angles), 1), axis=1)

    @staticmethod
    def intersection2lines(line_params, laserpoint, robotpos):
        a, b, c = line_params
        v_rot = np.array([a, b])  # 90° rotated directional vector of the line
        ba = laserpoint - robotpos  # directional vector of the laser scan
        # two parallel lines intersect in infinity
        return robotpos + ba * (-c - np.dot(robotpos, v_rot)) / \
            np.expand_dims(np.dot(ba, v_rot), len(laserpoint.shape) - 1) \
            if np.all(np.dot(ba, v_rot) != 0) else np.full(laserpoint.shape, np.inf)

    @staticmethod
    def odr_fit(data):  # orthogonal distance regression; data is an (n x d) matrix
        data_mean = np.mean(data, axis=0)
        _, _, V = np.linalg.svd(data - data_mean)  # singular value decomposition
        return -V[1, 0] * -1e4, V[1, 1] * 1e4, np.dot(data_mean, V[1]) * -1e4  # numbers a, b, c mustn't get too small

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
        for i in range(break_point_ind, (self.NP - self.PMIN), self.STEP):
            j = i + self.SNUM  # SNUM = Number of points in our seed segment
            if not np.all(np.linalg.norm(self.LASERPOINTS[i:j-1] - self.LASERPOINTS[i+1:j], axis=1) < self.Kappa):
                continue  # some could be skipped
            params = self.odr_fit(self.LASERPOINTS[i:j])
            predicted_points = self.intersection2lines(params, self.LASERPOINTS[i:j], robot_position)

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
        PB, PF = i, j

        while self.dist_point2line(line_eq, self.LASERPOINTS[PF]) < self.EPSILON and \
                np.linalg.norm(self.LASERPOINTS[PF] - self.LASERPOINTS[PF - 1]) < self.Kappa:
            if PB <= PF <= self.NP - 1:
                line_eq = self.odr_fit(self.LASERPOINTS[PB:PF])
            else:
                break
            PF += 1
        PF -= 1

        while self.dist_point2line(line_eq, self.LASERPOINTS[PB]) < self.EPSILON and \
                np.linalg.norm(self.LASERPOINTS[PB] - self.LASERPOINTS[PB + 1]) < self.Kappa:
            if break_point <= PB <= PF:
                line_eq = self.odr_fit(self.LASERPOINTS[PB:PF])
            else:
                break
            PB -= 1
        PB += 1

        # if the line is long enough and contains enough points
        if (np.linalg.norm(self.LASERPOINTS[PB] - self.LASERPOINTS[PF]) >= self.LMIN) and (PF - PB > self.PMIN):
            self.LINE_PARAMS = line_eq
            self.LINE_SEGMENTS.append((self.LASERPOINTS[PB + 1], self.LASERPOINTS[PF - 1]))
            return [line_eq, PB, PF]
        else:
            return False

    def lineFeats2point(self):
        new_rep = []  # the new representation of tne features
        for feature in self.FEATURES:
            new_rep.append([feature[0], feature[1], self.projection_point2line(feature[0], np.zeros(2))])
        return new_rep


def landmark_association(landmarks):
    thresh = 10
    for l in landmarks:

        flag = False
        for i, Landmark in enumerate(Landmarks):
            dist = np.linalg.norm(l[2] - np.array(Landmark[2]))
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
    length1 = np.linalg.norm(seg1[0] - np.array(seg1[1]))
    length2 = np.linalg.norm(seg2[0] - np.array(seg2[1]))
    center1 = np.array((seg1[0][0] + seg1[1][0]) / 2, (seg1[0][1] + seg1[1][1]) / 2)
    center2 = np.array((seg2[0][0] + seg2[1][0]) / 2, (seg2[0][1] + seg2[1][1]) / 2)
    dist = np.linalg.norm(center1 - center2)
    if dist > (length1 + length2) / 2:
        return False
    else:
        return True

import numpy as np


class Particle(object):
    def __init__(self, weight, time_stamps, name=None):
        self.name = name
        self.weight = weight
        self.time_stamps = time_stamps

        self.n = len(time_stamps)
        self.avail = np.ones_like(self.time_stamps, dtype=np.bool)

    def reset(self):
        self.avail = np.ones_like(self.time_stamps, dtype=np.bool)

    def add_alignment(self, idx):
        self.avail[idx] = False

    def del_alignment(self, idx):
        self.avail[idx] = True

    def available_time_stamps(self):
        return self.time_stamps[self.avail]

    def nearest_token(self, time_stamp, threshold=float('inf')):
        distances = np.abs(self.time_stamps - time_stamp)
        large_number = np.max(distances) * 2
        distances = distances + (large_number * (1-self.avail))
        selected_idx = np.argmin(distances)
        distance = distances[selected_idx]
        if distance >= threshold:
            selected_idx = -1
        return selected_idx, distance

    def index_from_avail(self, idx):
        real_idx = np.where(self.avail)[0][idx]
        return real_idx

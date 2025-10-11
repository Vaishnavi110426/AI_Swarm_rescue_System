import numpy as np

class SwarmCoordinator:
    def __init__(self, num_drones, num_humans):
        self.num_drones = num_drones
        self.num_humans = num_humans

    def assign_targets(self, drone_positions, human_positions):
        """
        Assign each drone a human to rescue based on nearest distance.
        Returns a list of target indices for each drone.
        """
        targets = [-1] * self.num_drones
        assigned = set()
        for i, drone_pos in enumerate(drone_positions):
            min_dist = float('inf')
            target_idx = -1
            for j, human_pos in enumerate(human_positions):
                if j in assigned or human_pos[0] == -1:
                    continue
                dist = np.linalg.norm(np.array(drone_pos) - np.array(human_pos))
                if dist < min_dist:
                    min_dist = dist
                    target_idx = j
            targets[i] = target_idx
            if target_idx != -1:
                assigned.add(target_idx)
        return targets

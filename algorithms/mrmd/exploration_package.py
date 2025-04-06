"""
exploration_package.py

Classes of the MultiResolutionMultiRobot paper that discuss frameworks
that explore certain areas.

"""

import numpy as np

class ExplorationPackage:
    """Implements an area that needs to be explored with a certain resolution"""
    def __init__(self, x_min, x_max, y_min, y_max, step):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.step = step

    @staticmethod
    def generate_lawnmower_step(x_min, x_max, y_min, y_max, step):
        """Generates a horizontal lawnmower path"""
        current = [x_min, y_min]
        path = []
        path.append(current)
        while True:
            path.append([x_max, current[1]])
            path.append([x_max, current[1]+step])
            path.append([x_min, current[1]+step])
            current = [x_min, current[1]+2 * step]
            path.append(current)
            if current[1] + step > y_max:
                break
        path.append([x_max, current[1]])
        return np.array(path)


    def get_path(self, type):
        """Generates a lawnmower path of one of the four types"""
        if type == "top-left":
            self.path = ExplorationPackage.generate_lawnmower_step(self.x_min + self.step//2, 
                                                self.x_max - self.step//2, 
                                                self.y_min + self.step//2, 
                                                self.y_max - self.step//2, self.step)
            return self.path
        if type == "top-right":
            # obtained by flipping x with y and reversing them
            path = ExplorationPackage.generate_lawnmower_step(self.y_min + self.step//2, 
                                                self.y_max - self.step//2,
                                                self.x_min + self.step//2, 
                                                self.x_max - self.step//2, 
                                                 self.step)
            self.path = np.flip(path, axis=1)
            return self.path
        if type == "bottom-left":
            raise Exception("path bottom-left not implemented yet")
            path = ExplorationPackage.generate_lawnmower_step(self.x_min + self.step//2, 
                                                self.x_max - self.step//2, 
                                                self.y_min + self.step//2, 
                                                self.y_max - self.step//2, self.step)        
        if type == "bottom-right":
            path = ExplorationPackage.generate_lawnmower_step(self.x_min + self.step//2, 
                                                self.x_max - self.step//2, 
                                                self.y_min + self.step//2, 
                                                self.y_max - self.step//2, self.step)
            self.path = np.flip(path, axis=0)
            return self.path
        
    
    

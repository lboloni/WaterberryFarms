
# allow the import from the source directory
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np

from InformationModel import AbstractScalarFieldIM, im_score, im_score_weighted, im_score_weighted_asymmetric
from environment import ScalarFieldEnvironment

from sklearn.metrics import accuracy_score

class testInformationModel(unittest.TestCase):

    def setUp(self):
        pass

    def test_(self):
        error = np.array([[0.1, -0.1], [0.05, 0.05]])
        mask = np.array([[1, 1], [0, 1]])
        env = ScalarFieldEnvironment("TestEnv", 2, 2, seed = 0)
        env.value.fill(0.5)
        im = AbstractScalarFieldIM("TestIM", 2, 2)
        im.value = env.value + error
        print(im.value)
        score1 = im_score(im, env)
        score2 = im_score_weighted(im, env, mask)
        score3 = im_score_weighted_asymmetric(im, env, 1, 100, mask)
        print(f"score: {score1} masked {score2} masked and weighted {score3}")

if __name__ == '__main__':
    y_pred = [0, 1, 2, 3]
    y_true = [0, 1, 2, 3]
    asc = accuracy_score(y_true, y_pred)
    print(f"as = {asc}")

    unittest.main()


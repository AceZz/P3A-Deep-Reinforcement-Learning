import numpy as np

def convert_action(action, binary_action = False):
    """TEMPORARY FUNCTION, NEED TO GO TO CONTINUOUS ACTIONS"""
    if not binary_action:
        if action == 0:
            return np.array([0, 0.2]) #buy stocks with 20% of remaining balance
        elif action == 1:
            return np.array([1, 0.2]) #sell 20% of stocks
        elif action == 2:
            return np.array([2, 0]) #do nothing
    else:
        if action == 0:
            return np.array([0, 0.2]) #buy stocks with 20% of remaining balance
        elif action == 1:
            return np.array([1, 0.2]) #sell 20% of stocks

import numpy as np


def d_dt(steps, values):
    slope = np.zeros(len(steps))

    # Compute Slope and Curvature
    for i in range(1, len(steps) - 1):
        dx1 = steps[i+1] - steps[i]
        dx0 = steps[i] - steps[i-1]
        
        dy1 = values[i+1] - values[i]
        dy0 = values[i] - values[i-1]
        
        slope[i] = (dy1 / dx1 + dy0 / dx0) / 2

    slope[0] = slope[1]
    slope[-1] = slope[-2]

    return slope

def d_dlogt(steps, values):
    slope = np.zeros(len(steps))
    log_steps = np.log(np.array(steps) + 1)

    # Compute Slope and Curvature
    for i in range(1, len(steps) - 1):
        dx1 = log_steps[i+1] - log_steps[i]
        dx0 = log_steps[i] - log_steps[i-1]
        
        dy1 = values[i+1] - values[i]
        dy0 = values[i] - values[i-1]
        
        slope[i] = (dy1 / dx1 + dy0 / dx0) / 2

    slope[0] = slope[1]
    slope[-1] = slope[-2]

    return slope

def dlog_dlogt(steps, values):
    slope = np.zeros(len(steps))
    log_steps = np.log(np.array(steps) + 1)
    log_values = np.log(values)

    # Compute Slope and Curvature
    for i in range(1, len(steps) - 1):
        dx1 = log_steps[i+1] - log_steps[i]
        dx0 = log_steps[i] - log_steps[i-1]
        
        dy1 = log_values[i+1] - log_values[i]
        dy0 = log_values[i] - log_values[i-1]
        
        slope[i] = (dy1 / dx1 + dy0 / dx0) / 2

    slope[0] = slope[1]
    slope[-1] = slope[-2]

    return slope
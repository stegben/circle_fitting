import numpy as np
from scipy.optimize import least_squares


def circle_residuals(params, x, y):
    """
    Calculate the residuals (errors) between the data points and the circle.
    params: (cx, cy, r) - circle center (cx, cy) and radius r
    x, y: arrays of data points coordinates
    """
    cx, cy, r = params
    return np.sqrt((x - cx)**2 + (y - cy)**2)-r


def fit_circle(x, y, initial_guess=None):
    """
    Fit a circle to the given x, y points using the Least Squares Circle Fitting method.
    x, y: arrays of data points coordinates
    initial_guess: initial estimation of the circle (optional)
    return: (cx, cy, r) - circle center (cx, cy) and radius r
    """
    if initial_guess is None:
        # Choose the first three points as the initial estimation
        initial_guess = (x.mean(), y.mean(), np.sqrt((x[0]-x.mean())**2 + (y[0]-y.mean())**2))

    # Use least_squares to perform the iterative nonlinear least squares optimization
    result = least_squares(circle_residuals, initial_guess, args=(x, y))

    cx, cy, r = result.x

    return cx, cy, r


def fit_circle_ransac(x, y, num_iterations=10, threshold=0.05):
    best_inliers = None
    best_params = None

    for _ in range(num_iterations):
        # Randomly sample 3 data points to form an initial estimation of the circle
        n_samples = min(12, len(x))
        indices = np.random.choice(len(x), n_samples, replace=False)
        initial_guess = (x[indices[0]], y[indices[0]], np.sqrt((x[indices[1]]-x[indices[0]])**2 + (y[indices[1]]-y[indices[0]])**2))

        # Fit the circle using the initial estimation
        if best_params is None:
            cx, cy, r = fit_circle(x, y, initial_guess)
        else:
            cx, cy, r = fit_circle(x, y, best_params)

        # Calculate residuals and identify inliers
        residuals = circle_residuals((cx, cy, r), x, y)
        inliers = np.abs(residuals) < threshold

        # Update best parameters if we found more inliers
        if (best_inliers is None) or (np.sum(inliers) > np.sum(best_inliers)):
            best_inliers = inliers
            best_params = (cx, cy, r)

     # Check if we have at least three inliers before refitting the circle
    if np.sum(best_inliers) >= 3:
        cx, cy, r = fit_circle(x[best_inliers], y[best_inliers])
    else:
        cx = x.mean()
        cy = y.mean()
        r = 0
    if (r > 2.8) or (r < 1.8):
        cx = x.mean()
        cy = y.mean()
    return cx, cy, r, best_inliers
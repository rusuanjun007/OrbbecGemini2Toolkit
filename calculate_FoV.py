import numpy as np


def h_fov(Z):
    """
    Calculate the field of view (FOV) based on the given parameters.

    Parameters:
    Z (float): The distance from the camera to the object.

    Returns:
    float: The calculated field of view.
    """
    cx = 636.404663
    fx = 613.787292
    B = 0.05
    width = 1280

    # Calculate the field of view using the provided formula
    # Note: The formula is derived from the pinhole camera model
    # and assumes a rectangular image sensor.

    # Ensure Z is not zero to avoid division by zero
    if Z == 0:
        raise ValueError("Z must be non-zero to calculate FOV.")

    # Calculate FOV
    # The formula is derived from the pinhole camera model
    # and assumes a rectangular image sensor.
    fov = np.arctan(cx / fx - B / Z) + np.arctan((width - 1 - cx) / fx)
    active_fov = np.arctan(cx / fx) + np.arctan((width - 1 - cx) / fx)

    # Convert radians to degrees
    return fov * 180 / np.pi, active_fov * 180 / np.pi


def v_fov(Z):
    """
    Calculate the vertical field of view (FOV) based on the given parameters.

    Parameters:
    Z (float): The distance from the camera to the object.

    Returns:
    float: The calculated vertical field of view.
    """
    cy = 396.343628
    fy = 613.787292
    B = 0.05
    height = 800

    # Ensure Z is not zero to avoid division by zero
    if Z == 0:
        raise ValueError("Z must be non-zero to calculate FOV.")

    # Calculate FOV
    fov = np.arctan(cy / fy - B / Z) + np.arctan((height - 1 - cy) / fy)
    active_fov = np.arctan(cy / fy) + np.arctan((height - 1 - cy) / fy)

    # Convert radians to degrees
    return fov * 180 / np.pi, active_fov * 180 / np.pi


print(h_fov(2))
print(v_fov(2))

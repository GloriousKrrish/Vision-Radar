import pytest
import numpy as np
from visionradar.cv.calibration import HomographyCalibrator

def test_homography_exact_recovery():
    """
    Tests that a known affine/perspective transform mapping is recovered with zero error.
    """
    # Define 4 image points and corresponding known world road coordinates (in meters)
    image_pts = [(100.0, 400.0), (700.0, 400.0), (500.0, 150.0), (300.0, 150.0)]
    world_pts = [(0.0, 10.0), (12.0, 10.0), (12.0, 90.0), (0.0, 90.0)]

    calibrator = HomographyCalibrator(image_points=image_pts, world_points=world_pts)

    # Test point transformation round-trip
    for img_p, world_p in zip(image_pts, world_pts):
        est_x, est_y = calibrator.image_to_world(*img_p)
        assert abs(est_x - world_p[0]) < 1e-3, f"Expected X={world_p[0]}, got {est_x}"
        assert abs(est_y - world_p[1]) < 1e-3, f"Expected Y={world_p[1]}, got {est_y}"

        inv_u, inv_v = calibrator.world_to_image(world_p[0], world_p[1])
        assert abs(inv_u - img_p[0]) < 1e-2
        assert abs(inv_v - img_p[1]) < 1e-2

    rmse = calibrator.compute_reprojection_rmse()
    assert rmse < 1e-3, f"Reprojection RMSE should be zero, got {rmse}"

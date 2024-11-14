import numpy as np


def triangulation(
        camera_matrix: np.ndarray,
        camera_position1: np.ndarray,
        camera_rotation1: np.ndarray,
        camera_position2: np.ndarray,
        camera_rotation2: np.ndarray,
        image_points1: np.ndarray,
        image_points2: np.ndarray
) -> np.ndarray:
    # Convert camera position and rotation from world coordinates to the camera coordinate system
    def get_projection_matrix(camera_position, camera_rotation):
        # Transform the camera position to the camera coordinate system
        transformed_position = -camera_rotation.T @ camera_position
        # Combine rotation and translation into the extrinsic matrix
        extrinsic_matrix = np.hstack((camera_rotation.T, transformed_position.reshape(-1, 1)))
        # Compute the projection matrix
        return camera_matrix @ extrinsic_matrix

    # Projection matrices for each camera
    P1 = get_projection_matrix(camera_position1, camera_rotation1)
    P2 = get_projection_matrix(camera_position2, camera_rotation2)

    num_points = image_points1.shape[0]
    points_3D = []

    for i in range(num_points):
        x1, y1 = image_points1[i]
        x2, y2 = image_points2[i]

        # Set up the A matrix for Ax = 0
        A = np.array([
            (x1 * P1[2] - P1[0]),
            (y1 * P1[2] - P1[1]),
            (x2 * P2[2] - P2[0]),
            (y2 * P2[2] - P2[1])
        ])

        # Solve for the null space of A
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X /= X[3]  # Normalize by the homogeneous coordinate

        points_3D.append(X[:3])

    return np.array(points_3D)

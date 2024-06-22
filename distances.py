from typing import Union
import numpy as np
import torch


def find_cosine_distance(
    source_representation: torch.Tensor, test_representation: torch.Tensor
) -> np.float64:
    """
    This function calculates the cosine distance between two given representations.

    Parameters:
    source_representation (torch.Tensor): The first representation.
    test_representation (torch.Tensor): The second representation.

    Returns:
    np.float64: The cosine distance between the two representations.
    """
    source_representation = source_representation.detach().numpy()
    test_representation = test_representation.detach().numpy()

    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def find_euclidean_distance(
    source_representation: Union[np.ndarray, torch.Tensor],
    test_representation: Union[np.ndarray, torch.Tensor]
) -> np.float64:
    """
    This function calculates the Euclidean distance between two given representations.

    Parameters:
    source_representation (Union[np.ndarray, torch.Tensor]): The first representation.
    test_representation (Union[np.ndarray, torch.Tensor]): The second representation.

    Returns:
    np.float64: The Euclidean distance between the two representations.
    """
    source_representation = source_representation.detach().numpy()
    test_representation = test_representation.detach().numpy()

    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def l2_normalize(x: torch.Tensor) -> np.ndarray:
    """
    This function normalizes the input tensor using L2 normalization.

    Parameters:
    x (torch.Tensor): The input tensor.

    Returns:
    np.ndarray: The normalized tensor.
    """
    x = x.detach().numpy()
    return x / np.sqrt(np.sum(np.multiply(x, x)))


def find_distance(
    alpha_embedding: torch.Tensor,
    beta_embedding: torch.Tensor,
    distance_metric: str,
) -> np.float64:
    """
    This function calculates the distance between two embeddings based on the specified metric.

    Parameters:
    alpha_embedding (torch.Tensor): The first embedding.
    beta_embedding (torch.Tensor): The second embedding.
    distance_metric (str): The distance metric to use.

    Returns:
    np.float64: The distance between the two embeddings.

    Raises:
    ValueError: If an invalid distance metric is passed.
    """
    if distance_metric == "cosine":
        distance = find_cosine_distance(alpha_embedding, beta_embedding)
    elif distance_metric == "euclidean":
        distance = find_euclidean_distance(alpha_embedding, beta_embedding)
    elif distance_metric == "euclidean_l2":
        distance = find_euclidean_distance(
            l2_normalize(alpha_embedding), l2_normalize(beta_embedding)
        )
    else:
        raise ValueError("Invalid distance_metric passed - ", distance_metric)
    return distance


def find_threshold(distance_metric: str, reletive_face_area: float) -> float:
    """
    This function finds the threshold for a given distance metric and relative face area.

    Parameters:
    distance_metric (str): The distance metric to use.
    reletive_face_area (float): The relative face area.

    Returns:
    float: The threshold value.
    """
    thresholds = {"cosine": 0.68, "euclidean": 1.17, "euclidean_l2": 1.17} if reletive_face_area > 0.02 else\
                 {"cosine": 0.5, "euclidean": 1.0, "euclidean_l2": 1.0}

    threshold = thresholds.get(distance_metric, 0.4)
    return threshold
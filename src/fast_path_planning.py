import numpy as np
import fastpathplanning as fpp
import networkx as nx
import matplotlib.pyplot as plt
from format_logger import setup_logger
import logging

setup_logger(level="INFO")
logger = logging.getLogger(__name__)

def convert_to_fpp_format(B):
    """
    Convert your graph to FastPathPlanning's SafeSet format.
    
    Args:
        B: Set of Boxes
    Returns:
        S: SafeSet object
    """
    
    L = []
    U = []
    
    for box in B:
        diagonal = box["diagonal"]
        min_point = np.array(diagonal[0])
        max_point = np.array(diagonal[1])  
        L.append(min_point)
        U.append(max_point)

    logger.info(f'Converted {len(L)} boxes to FastPathPlanning format.')

    L = np.array(L)
    U = np.array(U)

    S = fpp.SafeSet(L, U, verbose=False)
    
    return S

def extract_points_from_bezier_path(bezier_path, num_points=21):
    """
    Extract points from a Bezier path.
    
    Args:
    bezier_path: BezierCurve object
    num_points: Number of points to sample along the path

    Returns:
    path_points: List of points along the Bezier path
    """
    
    path_points = []
    t_values = np.linspace(bezier_path.a, bezier_path.b, num_points)

    path_points = [list(bezier_path(t).T) for t in t_values]
    return path_points

def plan_with_fpp(S, p_init, p_term, T=100, alpha=[1.0], der_init={}, der_term={}, verbose=True):
    """
    Plan a path using FastPathPlanning's plan function.
    
    Args:
        S: SafeSet object
        p_init: Initial point (numpy array)
        p_term: Terminal point (numpy array)
        T: Total time for the trajectory
        alpha: List of cost coefficients for derivatives
        der_init: Dictionary of initial derivative conditions
        der_term: Dictionary of terminal derivative conditions
        verbose: Boolean flag for verbosity
    Returns:
        path: Planned trajectory
    """
    
    path = fpp.plan(S, p_init, p_term, T, alpha, der_init, der_term, verbose)
    
    path_points = []
    for bezier in path.beziers:
        path_points.extend(extract_points_from_bezier_path(bezier))

    return path_points
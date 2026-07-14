
import numpy as np
def Q_matrix (E_11, E_22,v_12, G_12):
    """
    This function calculates the Q matrix for a composite lamina given the engineering constants.
    
    Parameters:
    E_11 : float
        Young's modulus in the fiber direction (Pa)
    E_22 : float
        Young's modulus in the transverse direction (Pa)
    v_12 : float
        Poisson's ratio (dimensionless)
    G_12 : float
        Shear modulus (Pa)
        
    Returns:
    Q : numpy.ndarray
        The stiffness matrix Q (3x3)
    """
    
    # Calculate the elements of the Q matrix
    Q_11 = E_11 / (1 - v_12 * v_12 * E_22 / E_11)
    Q_12 = v_12 * E_22 / (1 - v_12 * v_12 * E_22 / E_11)
    Q_22 = E_22 / (1 - v_12 * v_12 * E_22 / E_11)
    Q_66 = G_12
    
    # Construct the Q matrix
    Q = np.array([[Q_11, Q_12, 0],
                  [Q_12, Q_22, 0],
                  [0, 0, Q_66]])
    
    return Q

def CTE_vetor(alpha_1, alpha_2):
    """
    This function calculates the coefficient of thermal expansion (CTE) vector for a composite lamina.
    
    Parameters:
    alpha_1 : float
        Coefficient of thermal expansion in the fiber direction (1/K)
    alpha_2 : float
        Coefficient of thermal expansion in the transverse direction (1/K)
        
    Returns:
    alpha : numpy.ndarray
        The CTE vector (3x1)
    """
    
    # Construct the CTE vector
    alpha = np.array([[alpha_1],
                      [alpha_2],
                      [0]])
    
    return alpha


def T_matrix(theta):
    """
    This function calculates the transformation matrix T for a given angle theta.
    
    Parameters:
    theta : float
        Angle in degrees
        
    Returns:
    T : numpy.ndarray
        The transformation matrix T (3x3)
    """
    
    # Convert angle from degrees to radians
    theta_rad = np.radians(theta)
    
    # Calculate the elements of the transformation matrix
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    
    T = np.array([[c**2, s**2, 2*c*s],
                  [s**2, c**2, -2*c*s],
                  [-c*s, c*s, c**2 - s**2]])
    
    return T
def reuters_matrix ():
    """
    This function calculates the Reuter's matrix for a composite lamina.
    
    Returns:
    R : numpy.ndarray
        The Reuter's matrix (3x3)
    """
    
    # Construct the Reuter's matrix
    R = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 2]])
    
    return R


def Q_bar (Q, T, R):
    """
    This function calculates the transformed stiffness matrix Q_bar for a composite lamina.
    
    Parameters:
    Q : numpy.ndarray
        The stiffness matrix Q (3x3)
    T : numpy.ndarray
        The transformation matrix T (3x3)
    R : numpy.ndarray
        The Reuter's matrix (3x3)
        
    Returns:
    Q_bar : numpy.ndarray
        The transformed stiffness matrix Q_bar (3x3)
    """
    
    # Calculate the transformed stiffness matrix Q_bar
    Q_bar = np.linalg.inv(T) @ Q @ R @ T @ np.linalg.inv(R)
    
    return Q_bar


def alpha_bar(alpha, T, R):
    """
    This function calculates the transformed coefficient of thermal expansion (CTE) vector alpha_bar for a composite lamina.
    
    Parameters:
    alpha : numpy.ndarray
        The CTE vector (3x1)
    T : numpy.ndarray
        The transformation matrix T (3x3)
    R : numpy.ndarray
        The Reuter's matrix (3x3)
        
    Returns:
    alpha_bar : numpy.ndarray
        The transformed CTE vector alpha_bar (3x1)
    """
    
    # Calculate the transformed CTE vector alpha_bar
    alpha_bar = R @ T @ np.linalg.inv(R) @ alpha
    
    return alpha_bar



def A_matrix (Q_bars, ply_number, thickness):
    """
    This function calculates the A matrix for a composite laminate given the transformed stiffness matrices of each ply.
    
    Parameters:
    Q_bars : list of numpy.ndarray
        List of transformed stiffness matrices Q_bar for each ply (3x3)
    ply_number : int
        Number of plies in the laminate
        
    Returns:
    A : numpy.ndarray
        The A matrix (3x3)
    """
    
    # Initialize the A matrix
    A = np.zeros((3, 3))
    
    # Calculate the A matrix by summing the contributions of each ply
    for i in range(ply_number):
        A += Q_bars[i] * thickness
    
    return A 


def B_matrix(Q_bars, ply_number, thickness):
    """
    This function calculates the B matrix for a composite laminate given the 
    transformed stiffness matrices of each ply, assuming constant ply thickness.
    
    Parameters:
    Q_bars : list of numpy.ndarray
        List of transformed stiffness matrices Q_bar for each ply (3x3)
    ply_number : int
        Number of plies in the laminate
    thickness : float
        Thickness of a single ply
        
    Returns:
    B : numpy.ndarray
        The B matrix (3x3)
    """
    
    # Initialize the B matrix
    B = np.zeros((3, 3))
    
    # 1. Calculate the total thickness of the laminate
    total_thickness = ply_number * thickness
    
    # 2. Find the true bottom coordinate (z = -h/2)
    z_bottom = -total_thickness / 2
    
    # Calculate the B matrix by summing the contributions of each ply
    for i in range(ply_number):
        z_i = z_bottom + i * thickness      # Bottom surface of the current ply
        z_ip1 = z_i + thickness             # Top surface of the current ply
        
        B += Q_bars[i] * (z_ip1**2 - z_i**2) / 2
    
    return B
def D_matrix(Q_bars, ply_number, thickness):
    """
    This function calculates the D matrix for a composite laminate given the 
    transformed stiffness matrices of each ply, assuming constant ply thickness.
    
    Parameters:
    Q_bars : list of numpy.ndarray
        List of transformed stiffness matrices Q_bar for each ply (3x3)
    ply_number : int
        Number of plies in the laminate
    thickness : float
        Thickness of a single ply
        
    Returns:
    D : numpy.ndarray
        The D matrix (3x3)
    """
    
    # Initialize the D matrix
    D = np.zeros((3, 3))
    
    # 1. Calculate the total thickness of the laminate
    total_thickness = ply_number * thickness
    
    # 2. Find the true bottom coordinate (z = -h/2)
    z_bottom = -total_thickness / 2
    
    # Calculate the D matrix by summing the contributions of each ply
    for i in range(ply_number):
        z_i = z_bottom + i * thickness      # Bottom surface of the current ply
        z_ip1 = z_i + thickness             # Top surface of the current ply
        
        D += Q_bars[i] * (z_ip1**3 - z_i**3) / 3
    
    return D


def arrange_ABD(A,B,D):
    """
    This function arranges the A, B, and D matrices into a single ABD matrix.
    
    Parameters:
    A : numpy.ndarray
        The A matrix (3x3)
    B : numpy.ndarray
        The B matrix (3x3)
    D : numpy.ndarray
        The D matrix (3x3)
        
    Returns:
    ABD : numpy.ndarray
        The combined ABD matrix (6x6)
    """
    
    # Initialize the ABD matrix
    ABD = np.zeros((6, 6))
    
    # Fill in the A, B, and D matrices into the ABD matrix
    ABD[0:3, 0:3] = A
    ABD[0:3, 3:6] = B
    ABD[3:6, 0:3] = B
    ABD[3:6, 3:6] = D
    
    return ABD


def N_thermal(alpha_bars, Q_bars, delta_T, thickness, ply_number):
    """
    This function calculates the thermal force vector N_thermal for a composite laminate.
    
    Parameters:
    alpha_bars : list of numpy.ndarray
        List of transformed CTE vectors alpha_bar for each ply (1D arrays of length 3)
    Q_bars : list of numpy.ndarray
        List of transformed stiffness matrices Q_bar for each ply (3x3)
    delta_T : float
        Temperature change (K)
    thickness : float
        Thickness of a single ply
    ply_number : int
        Number of plies in the laminate
        
    Returns:
    N_thermal : numpy.ndarray
        The thermal force vector N_thermal (1D array of length 3)
    """
    
    # Initialize the thermal force vector as a 1D array
    N_T = np.zeros(3)
    
    # Calculate the thermal force vector by summing the contributions of each ply
    for i in range(ply_number):
        # Q_bars[i] @ alpha_bars[i] gives the thermal stress for that specific ply
        N_T += Q_bars[i] @ alpha_bars[i] * delta_T * thickness
    
    return N_T


def M_thermal(alpha_bars, Q_bars, delta_T, thickness, ply_number):
    """
    This function calculates the thermal moment vector M_thermal for a composite laminate.
    
    Parameters:
    alpha_bars : list of numpy.ndarray
        List of transformed CTE vectors alpha_bar for each ply (1D arrays of length 3)
    Q_bars : list of numpy.ndarray
        List of transformed stiffness matrices Q_bar for each ply (3x3)
    delta_T : float
        Temperature change (K)
    thickness : float
        Thickness of a single ply
    ply_number : int
        Number of plies in the laminate
        
    Returns:
    M_thermal : numpy.ndarray
        The thermal moment vector M_thermal (1D array of length 3)
    """
    
    # Initialize the thermal moment vector as a 1D array
    M_T = np.zeros(3)
    
    # Calculate the total thickness of the laminate
    total_thickness = ply_number * thickness
    
    # Find the true bottom coordinate (z = -h/2)
    z_bottom = -total_thickness / 2
    
    # Calculate the thermal moment vector by summing the contributions of each ply
    for i in range(ply_number):
        z_i = z_bottom + i * thickness      # Bottom surface of the current ply
        z_ip1 = z_i + thickness             # Top surface of the current ply
        
        # Calculate moment contribution for this specific ply
        M_T += Q_bars[i] @ alpha_bars[i] * delta_T * (z_ip1**2 - z_i**2) / 2
    
    return M_T

def resultantvector(N_thermal, M_thermal, ABD):
    """
    This function calculates the resultant strain and curvature vector for a composite laminate.
    
    Parameters:
    N_thermal : numpy.ndarray
        The thermal force vector N_thermal (1D array of length 3)
    M_thermal : numpy.ndarray
        The thermal moment vector M_thermal (1D array of length 3)
    ABD : numpy.ndarray
        The combined ABD matrix (6x6)
        
    Returns:
    result : numpy.ndarray
        The resultant strain and curvature vector (1D array of length 6)
    """
    
    # Combine the thermal force and moment vectors into a single vector
    NM = np.concatenate((N_thermal, M_thermal))
    
    # Calculate the resultant strain and curvature vector by multiplying with the inverse of ABD
    result = np.linalg.inv(ABD) @ NM
    
    return result       
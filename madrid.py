import numpy as np
from numba import njit
from typing import Optional
from damp import DAMP_v2, MASS

@njit
def MADRID(T: np.ndarray, train_test_split: int,
           minL: Optional[int] = 8,
           maxL: Optional[int] = None,
           stepsize: Optional[int]=1,
           subseqence_range = "Auto"):
    """
    MADRID algorithm, discovering time series discords of All Lengths.
     
    Parameters
    ----------
    T : np.array
        1D time series data of shape (n,)
    train_test_split : int
        Location of split point between training and test data
    minL: int
        min value of subsequence length to be searched
    maxL: int
        max value of subsequence length to be searched
    stepsize: int
        stepsize of subsequence lengths candidate set
    Returns
    -------
    best_discord_loc: tupple(int,int)
        start-end points of best discord 
    M: np.ndarray
        Multi-length discord table, (number of m, n)
    BSF_m: np.array
        Best So Far discord (normalized) value by m
    BSF_loc_m: np.array
        Location point of BSF by m
    m_set: np.array
        subsequence lengths candidate set
    """
    
    if maxL is None:
        maxL = train_test_split // 20
    if subseqence_range == "Auto":    
        m_set = np.unique(np.linspace(8, maxL + 1, 50).astype(np.int64))
    else:
        m_set = np.arange(minL, maxL + 1, stepsize, dtype=np.int64)
    n = len(T)
    m_range = len(m_set)
    M = np.zeros((m_range, n))
    BSF_m = np.zeros(m_range, dtype=float)
    BSF_loc_m = np.zeros(m_range, dtype=np.int64)
    is_calculated_flags = np.zeros(m_range, dtype=np.bool_)
    M, BSF_m, BSF_loc_m, is_calculated_flags = warm_up_phase(T,
                                                             train_test_split,
                                                             M,
                                                             BSF_m, BSF_loc_m, m_set,
                                                             is_calculated_flags)
    M, BSF_m, BSF_loc_m, is_calculated_flags = main_phase(T,
                                                          train_test_split,
                                                          M,
                                                          BSF_m, BSF_loc_m, m_set,
                                                          is_calculated_flags)
    best_m_pointer = np.argmax(BSF_m)
    best_discord_loc = (BSF_loc_m[best_m_pointer], BSF_loc_m[best_m_pointer]+m_set[best_m_pointer])
    return best_discord_loc, M, BSF_m, BSF_loc_m, m_set


@njit
def warm_up_phase(T: np.ndarray, train_test_split: np.ndarray, M: np.ndarray,
                  BSF_m: np.ndarray, BSF_loc_m: np.ndarray, 
                  m_set: np.ndarray, is_calculated_flags: np.ndarray):
    """
    warm_up_phase in MADRID algorithm.
    
    Parameters
    ----------
    T : np.ndarray
        1D time series data of shape (n,)
    train_test_split : np.ndarray
        Location of split point between training and test data

    Returns
    -------
    M: np.ndarray
        Multi-length discord table, (number of m, n)
    BSF_m: np.ndarray
        Best So Far discord (normalized) value by m
    BSF_loc_m: np.ndarray
        Location point of BSF by m
    is_calculated_flags: np.ndarray
        flags whether subsequence T[i:i+m] is a BSF candidate or not.
    """
    warmup_m_points = np.array([len(m_set)//2, 0, len(m_set)-1])
    for m_pointer_warm_up in warmup_m_points:
        m_warm_up = m_set[m_pointer_warm_up]
        discord_score, Left_MP = DAMP_v2(T, m_warm_up, train_test_split,
                                         np.sqrt(m_warm_up)*M[m_pointer_warm_up, :len(T)-m_warm_up+1], 
                                         np.sqrt(m_warm_up)*BSF_m[m_pointer_warm_up])
        BSF_m[m_pointer_warm_up] = discord_score * (1/(np.sqrt(m_warm_up)))
        M[m_pointer_warm_up, :len(T)-m_warm_up+1] = Left_MP * (1/(np.sqrt(m_warm_up)))
        BSF_loc_temp = np.argmax(Left_MP)
        BSF_loc_m[m_pointer_warm_up] = BSF_loc_temp
        is_calculated_flags[m_pointer_warm_up] = True
        for m_pointer in range(len(m_set)):
            if is_calculated_flags[m_pointer] == True:
                continue
            m = m_set[m_pointer]
            query = T[BSF_loc_temp:BSF_loc_temp+m]
            discord_score = np.nanmin(np.real(MASS(T[:BSF_loc_temp], query))) * (1/(np.sqrt(m)))
            M[m_pointer, BSF_loc_temp] = discord_score
            if discord_score == np.inf:
                print("m",m,"discord_score",discord_score)
            if BSF_m[m_pointer] < discord_score:
                BSF_m[m_pointer] = discord_score
                BSF_loc_m[m_pointer] = BSF_loc_temp
    return M, BSF_m, BSF_loc_m, is_calculated_flags


@njit
def main_phase(T: np.ndarray, train_test_split: np.ndarray,
               M: np.ndarray, BSF_m: np.ndarray,
               BSF_loc_m: np.ndarray, m_set: np.ndarray,
               is_calculated_flags: np.ndarray):
    """
    main roop in MADRID algorithm.
    
    Parameters
    ----------
    T : np.ndarray
        1D time series data of shape (n,)
    train_test_split : np.ndarray
        Location of split point between training and test data

    Returns
    -------
    M: np.ndarray
        Multi-length discord table, (number of m, n)
    BSF_m: np.ndarray
        Best So Far discord (normalized) value by m
    BSF_loc_m: np.ndarray
        Location point of BSF by m
    is_calculated_flags: np.ndarray
        flags whether subsequence T[i:i+m] is a BSF candidate or not.
    """
    for m_pointer in range(len(m_set)):
        if is_calculated_flags[m_pointer] == True:
            continue
        m = m_set[m_pointer]
        discord_score, Left_MP = DAMP_v2(T, m, train_test_split,
                                          np.sqrt(m)*M[m_pointer, :len(T)-m+1], np.sqrt(m)*BSF_m[m_pointer])
        M[m_pointer, :len(T)-m+1] = Left_MP * (1/(np.sqrt(m)))
        BSF_loc_temp = np.argmax(Left_MP)
        BSF_m[m_pointer] = discord_score * (1/(np.sqrt(m)))
        BSF_loc_m[m_pointer] = BSF_loc_temp
        is_calculated_flags[m_pointer] = True
    return M, BSF_m, BSF_loc_m, is_calculated_flags

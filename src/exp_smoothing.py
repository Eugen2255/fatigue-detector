import numpy as np
from numpy.typing import NDArray



def exp_smoothing(
        smooth_kps: NDArray[np.int32],
        raw_kps: NDArray[np.int32],
        coef: float = 0.3
) -> NDArray[np.int32]:

    '''    
        Экспоненциальное сглаживание координат точек
        
    Args:
        smooth_kps (NDArray[np.int32]): предыдущие СГЛАЖЕННЫЕ точки и их координаты
                                       shape(num_kp, 2)
                                       num_kp - кол-во ключевых точек
                                       2 - координаты (x,y)
        raw_kps (NDArray[np.int32]): новые СЫРЫЕ точки и их координаты, shape(num_kp, 2)
        coef (float) = 0.3: коэффициент регулярицазации [0..1]. Чем ближе к 0, тем больше сглаживание

    Returns:
        NDArray[np.int32]: новые актуальные сглаженные точки

    Note:
        используется для ОДНОГО списка точек (определенного лица, позы). Вызывать на каждой итерации !!
    '''

    return (coef * raw_kps + (1 - coef) * smooth_kps).astype(np.int32)
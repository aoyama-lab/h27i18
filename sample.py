import numpy as np
import tkinter as Tk
from sklearn.metrics import jaccard_similarity_score
import matplotlib.pyplot as plt

def jaccard(x,y):
    #集合表記 |A∧B|/|A∨B|
    N=np.dot(x,y)
    D=np.linalg.norm(x)+np.linalg.norm(y)-N
    print(N/D)


def dice():
    '''
    2|A∧B|/|A|+|B|
    分母が共通要素を無視して足しているため分子に二倍
    '''
def simpson():
    '''
    |A∧B|/小さい方(|A|,|B|)
    ただし、A∈Bのとき、simは1でもjacは1でないときがある。
    真部分集合を考慮する必要性がある。
    '''





if __name__ == '__main__':
    vec_a=np.array([0,1,0])
    vec_b=np.array([0,1,0])
    jaccard(vec_a,vec_b)
    x = np.arange(-3, 3, 0.1)
    y = np.sin(x)
    plt.plot(x, y)

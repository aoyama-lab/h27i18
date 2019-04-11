import numpy as np
from sklearn.metrics import jaccard_similarity_score
import matplotlib.pyplot as plt

def jaccard(x,y):
    #ベクトル表記　Va・Vb / ( |Va|^2 + |Vb|^2 - Va・Vb )
    result=jaccard_similarity_score(x,y)
    print(result)



if __name__ == '__main__':
    vec_a=['1','0','0']
    vec_b=['1','0','1']
    jaccard(vec_a,vec_b)
    x = np.arange(-3, 3, 0.1)
    y = np.sin(x)
    plt.plot(x, y)

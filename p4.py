import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('./ToyotaCorolla.csv')
plt.boxplot(df[['Price','HP','KM']])
plt.xticks([1,2,3],['Price','HP','KM'])
plt.show()

def minimax(depth, alpha, beta, nodeIndex, maxP, values):
    if depth == 3:
        return values[nodeIndex]
    if maxP:
        best = float('-inf')
        for i in range(2):
            val = minimax(depth+1, alpha, beta, nodeIndex*2+i, False, values)
            best = max(best,val)
            alpha = max(best, alpha)
            if beta<=alpha:
                break
            
        return best
    else:
        best = float('inf')
        for i in range(2):
            val = minimax(depth+1, alpha, beta, nodeIndex*2+i, True, values)
            best = min(best,val)
            beta = min(best, beta)
            if beta<=alpha:
                break
        return best
values = [3,5,2,9,12,5,23,23]
print(minimax(0,float('-inf'),float('inf'),0,True,values))

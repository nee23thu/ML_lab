import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('./ToyotaCorolla.csv')
sns.heatmap(df[['Price','KM','Weight','Doors']].corr(),cmap='jet')
plt.show()


def minimax(depth, nodeIndex, maxP, values):
    if depth == 3:
        return values[nodeIndex]
    if maxP:
        best = float('-inf')
        for i in range(2):
            val = minimax(depth+1, nodeIndex*2+i, False, values)
            best = max(best,val)
        return best
    else:
        best = float('inf')
        for i in range(2):
            val=minimax(depth+1,nodeIndex*2+i,True,values)
            best=min(best,val)
        return best
values = [3,5,2,9,12,5,23,23]
print(minimax(0,0,True,values))

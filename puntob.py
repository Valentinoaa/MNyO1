import pandas as pd
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

def main():
    inter = pd.read_csv('Mediciones/mnyo_mediciones.csv', header=None, sep='\s+', float_precision='high')

    inter.columns = ['column1', 'column2']

    
    plt.plot(inter['column1'], inter['column2'], color='black')
    plt.xlabel('Column 1')
    plt.ylabel('Column 2')

    grid = scipy.interpolate.interp1d(inter['column1'], inter['column2'], kind='linear')
    print(inter['column1'])
    interval = np.array([])

    for i in range(0, len(inter['column1']) - 1):
    
        interval = np.append(interval, np.linspace(inter['column1'][i], inter['column1'][i+1], 3))
        
    interval = np.unique(interval)

    plt.plot(interval, grid(interval))


    
    plt.title('Plot of Column 1 vs Column 2')
    plt.show()
    
if __name__ == "__main__":
    main()
    
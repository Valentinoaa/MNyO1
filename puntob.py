import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv('Mediciones/mnyo_mediciones.csv', header=None, sep=r'\s+')
    df.columns = ['column1', 'column2']
    
    plt.plot(df['column1'], df['column2'])
    plt.xlabel('Column 1')
    plt.ylabel('Column 2')
    plt.title('Plot of Column 1 vs Column 2')
    plt.show()
    
if __name__ == "__main__":
    main()
    
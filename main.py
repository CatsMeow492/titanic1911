import numpy as np # linear algebra
import pandas as pd
from pandas.core.indexes.base import InvalidIndexError # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns; sns.set_theme() 
import matplotlib.pyplot as plt
from math import log2
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os 

            
training = pd.read_csv('./csv/train.csv')
test = pd.read_csv('./csv/test.csv')

training['train_test'] = 1
test['train_test'] = 0

test['Survived'] = np.NAN 
all_data = pd.concat([training, test])

# Create a data exploration file and write the columns to the file as well as the training and test set info

f = open("dataexploration.txt", 'w')
f.write('Columns' + '\n' + '----------------------------------------------------------------------' + '\n')
f.write(str(all_data.columns))
f.close()
f = open('dataexploration.txt', 'a')
f.write('\n'+'----------------------------------------------------------------------' + '\n' + 'Training Info' + '\n' + '----------------------------------------------------------------------' + '\n')
f.write(str(training.info))
f.write('\n'+'----------------------------------------------------------------------' + '\n' + 'Training Description' + '\n' + '----------------------------------------------------------------------' + '\n')
f.write(str(training.describe()))
f.write('\n'+'----------------------------------------------------------------------' + '\n' + 'Test Info' + '\n' + '----------------------------------------------------------------------' + '\n')
f.write(str(test.info))
f.write('\n'+'----------------------------------------------------------------------' + '\n' + 'Test Description' + '\n' + '----------------------------------------------------------------------' + '\n')
f.write(str(test.describe()))
f.close()
f = open("dataexploration.txt", 'r')
f.close()

# look at numeric and categorical values separately 
df_num = training[['Age','SibSp','Parch','Fare']]
df_cat = training[['Survived','Pclass','Sex','Ticket','Cabin','Embarked']]

# distributions for all numeric variables 
# iterate through all columns, create a plot, and output to /figures/

for i in df_num.columns:
    plt.hist(df_num[i])
    plt.title(i)
    plt.savefig(f'./figures/{i}.png')
    plt.close()
    
# Create heatmap of variables and save to /figures/   
    
print(df_num.corr())    
heatmap = sns.heatmap(df_num.corr(), annot = True, linewidths=.5)
fig = heatmap.get_figure()
fig.savefig('./figures/Heatmap.png')

for i in df_cat.columns:
    sns.barplot(df_cat[i].value_counts().index,df_cat[i].value_counts()).set_title(i)
    plt.savefig(f'./figures/{i}.png')
    plt.close()
    
# Here we get into information gain

#function for calculating entropy
def entropy(class0, class1):
    return (class0 * log2(class0) + class 1 * log2(class1))





            




# import seaborn library
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_excel('results.xlsx', sheet_name=10)

# print(dataset.head())


# load the dataset
# data = sns.load_dataset('tips')

# # view the dataset
# print(data.head(5))

data = dataset.query('Sampling == 0 and Metric == "F1"')
# # data = dataset
# data = data.groupby(['Method', 'Metric', 'Sampling']).agg({'Value': ['mean']})
# print(data.head(1000000))


# data_bc = dataset.query('Sampling == 1')

sns.boxplot(x = data['Method'],
            y = data['Value'],
            hue = data['Classifier'],
            palette = 'Greys_r')




# RQ3: multiclass
# data = dataset.query('Metric == "F1" and n == 64') 
# data = dataset.query('Metric == "F1" and n == 64') 
# sns.boxplot(x = data['Method'],
#             y = data['Value'],
#             hue = data['Sampling'],
#             palette = 'Greys_r')


# data = dataset.query('Sampling == 1')
# data = data.query('Metric == "AUC"')
# sns.boxplot(x = data['Method'],
#             y = data['Value'],
#             hue = data['n'],
#             palette = 'Greys')

# data = dataset.query('Sampling == 0')
# data = data.query('Metric == "AUC"')
# sns.boxplot(x = data['Method'],
#             y = data['Value'],
#             hue = data['n'],
#             palette = 'Greys')

plt.show()
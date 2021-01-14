from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import matplotlib.pyplot as plt
import sys

data = pd.read_csv('data_2019.csv')[['launch_angle', 'launch_speed', 'woba_value']]
data = data.dropna()
X = data[['launch_speed','launch_angle']].values
y = data['woba_value'].values

neigh = KNeighborsRegressor(n_neighbors=400)
neigh.fit(X, y)

output = []
i=0
for x in X:
	print(f'\r{i}/{len(X)}', end='')
	sys.stdout.flush()
	output.append(neigh.predict([x])[0])
	i+=1

cm = plt.cm.get_cmap('RdYlBu')
sc = plt.scatter(X[:,0], X[:,1], c=output, s=1, cmap=cm)
plt.colorbar(sc)
plt.show()
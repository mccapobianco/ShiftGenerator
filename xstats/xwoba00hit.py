from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pickle

re_matrix = pd.read_csv('re_matrix.csv')
v_out = re_matrix['1 Out'][0]
v_single = re_matrix['0 Outs'][1]-v_out
v_double = re_matrix['0 Outs'][2]-v_out
v_triple = re_matrix['0 Outs'][4]-v_out
v_homerun = re_matrix['0 Outs'][0]-v_out+1

data = pd.read_csv('data_2019.csv')[['launch_angle', 'launch_speed', 'woba_value']]
data = data.dropna()
conversions = {0:0, 0.7:v_single, 0.9:v_single, 1.25:v_double, 1.6:v_triple, 2:v_homerun}
data['woba00_value'] = data['woba_value'].apply(lambda x : conversions[x])

data = data[data['woba_value'] != 0]
X = data[['launch_speed','launch_angle']].values
y = data['woba00_value'].values

neigh = KNeighborsRegressor(n_neighbors=250)
neigh.fit(X, y)

with open('woba00_model.pkl', 'wb') as f:
	pickle.dump(neigh, f)
	
output = []
i=0
# for x in X:
# 	print(f'\r{i}/{len(X)}', end='')
# 	sys.stdout.flush()
# 	output.append(neigh.predict([x])[0])
# 	i+=1
output = neigh.predict(X)

cm = plt.cm.get_cmap('RdYlBu')
sc = plt.scatter(X[:,0], X[:,1], c=output, s=1, cmap=cm)
plt.colorbar(sc)
plt.show()

I = []
J = []
output = []
for i in range(20,120):
	for j in range(-90,80):
		I.append(i)
		J.append(j)
		output.append(neigh.predict([[i,j]])[0])
cm = plt.cm.get_cmap('RdYlBu')
sc = plt.scatter(I, J, c=output, s=1, cmap=cm)
plt.colorbar(sc)
plt.show()
import pybaseball
import pandas as pd

YEAR = 2019

end_days = {3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30}
dfs = []
for month in range(3,10):
	data = pybaseball.statcast(f'{YEAR}-0{month}-01', f'{YEAR}-0{month}' + '-%02d'%end_days[month])
	data = data[data['description'].apply(lambda x: 'hit_into_play' in x)]
	dfs.append(data)

data = pd.concat(dfs)
data.to_csv(f'data_{YEAR}.csv', index=False)
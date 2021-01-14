import pybaseball

def data_from_name(last, first, year1=2020, num_years=1):
	years = range(year1, year1+num_years)
	lookup = pybaseball.playerid_lookup(last, first)
	if len(lookup) > 1:
		print('Multiple players found, determining player by years.')
		lookup['int'] = lookup.apply(lambda row: len(set(range(int(row['mlb_played_first']), int(row['mlb_played_last'])))&set(years)), axis=1)
		lookup = lookup[lookup['int']==max(lookup['int'])]
	if len(lookup) > 1:
		print('Unable to determine player')
	else:
		mlb_id = int(lookup['key_mlbam'])
		data = pybaseball.statcast_batter(f'{year1}-01-01', f'{year1+num_years}-01-01', mlb_id)
		data = data[data.apply(lambda row: 'hit_into_play' in row['description'], axis=1)]
		data = data[data['events'] != 'home_run']
		data = data.dropna(how='any', subset=['launch_angle', 'launch_speed', 'hc_x', 'hc_y'])
		return data

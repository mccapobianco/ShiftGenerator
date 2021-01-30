from shift import centroid_adjust, gradient_descent, process_hits, evaluate_alignment, fielders
from load_data import data_from_name
from display_field import draw_field
import sys
from random import shuffle

if __name__ == '__main__':
	name = []
	year_info = '2019'
	num_years = 1
	if len(sys.argv) < 2:
		sys.exit('Command must follow syntax: `python shift.py NAME [{-y | --year} YEAR [NUM_YEARS]]`')
	if len(sys.argv) > 1:
		args = ' '.join(sys.argv[1:])
		if '--year' in args:
			name, year_info = [x.strip() for x in args.split('--year')]
		elif '-y' in args:
			name, year_info = [x.strip() for x in args.split('-y')]
		else:
			name = args
		name = name.split(',')
		year_info = year_info.split(' ')
		if len(year_info) > 1:
			year1, num_years = [int(x) for x in year_info]
		else:
			year1 = int(year_info[0])
	data = data_from_name(name[0], name[1], year1=year1, num_years=num_years)
	data = process_hits(data, ignore_foul=True)
	hit_dots = [hit['hit_coord'] for hit in data]
	### TESING SPLIT
	shuffle(data)
	test_split = 1/3
	split = int(test_split*len(data))
	test_data = data[:split]
	train_data = data[split:]

	standard_value = evaluate_alignment(test_data)
	for _ in range(5):
		centroid_adjust(train_data, epochs=10, weight=1/2)
		gradient_descent(train_data, epochs=5, if_lr=1e4, of_lr=1e4, display=False)
	gradient_descent(train_data, epochs=15, if_lr=1e4, of_lr=1e4, display=False)
	player_dots = [tuple(player['loc'].numpy()) for player in fielders]
	shifted_value = evaluate_alignment(test_data)
	print(player_dots)
	print('Standard Value', standard_value)
	print('Shifted Value', shifted_value)
	draw_field(330, 420, players=player_dots, balls=hit_dots,title=f'Generated shift for {name[1]} {name[0]}'.upper())


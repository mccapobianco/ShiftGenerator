import sys
if len(sys.argv) < 2:
	sys.exit('Command must follow syntax: `python shift.py BATTER_NAME [{-y | --year} START_YEAR [NUM_YEARS]]`')

import tensorflow as tf
from display_field import draw_field
import pickle
import json
import math
import numpy as np
import pandas as pd
from load_data import data_from_name

with open('xstats/woba00_model.pkl', 'rb') as f:
	woba_model = pickle.load(f)
with open('fielders.json', 'r') as f:
	fielders = json.load(f)
for player in fielders:
	player['loc'] = tf.Variable([player['x'], player['y']], dtype=float)


def sigmoid(center=0, horz_compress=1):
	return lambda x: 1 / (1 + tf.math.exp(-horz_compress*(x-center)))

def sinemoid(center=0, horz_compress=1): #a function similar to sigmoid but bound between 0 and 1 inclusively
	return lambda x: 1 if x > center+1/horz_compress else 0 if x < center-1/horz_compress else 0.5+0.5*tf.math.sin((x-center)*math.pi/2*horz_compress)

def est_hangtime(la, ev, d, bh):
	# v0 = ev*tf.math.sin(la*math.pi/180)
	# return 0.3779595 * (-v0 - tf.math.sqrt(v0**2+60))/-9.8
	lam = lambda v0 : (-v0 - np.sqrt(v0**2+60))/-9.8
	ht = lam(ev*math.sin(la*math.pi/180))**(0.7-0.05*abs(d*math.pi/180))
	return max(.9847*ht-0.16117,tf.Variable(0.0))

def est_height(hangtime):
	g = 32.1740 # ft/s^2
	v0 = g*hangtime/2-3/hangtime
	return lambda t : 3+v0*t-1/2*g*t**2

	
def process_hits(df, ignore_foul=False, output_file=None): #turn df into list of hits (dicts)
	hits = []
	df['weight'] = woba_model.predict(df[['launch_speed', 'launch_angle']])
	for index, row in df.iterrows():
		launch_angle = row['launch_angle']
		exit_velo = row['launch_speed']
		hit_coord = (2.75*(row['hc_x'] - 124), 2.75*(204 - row['hc_y']))
		dir_rad = tf.math.atan(hit_coord[0]/hit_coord[1])
		est_dir = (dir_rad*180/math.pi)/2
		hangtime = est_hangtime(launch_angle, exit_velo, est_dir, row['stand'])
		height = est_height(hangtime)
		hit_dist = row['hit_distance_sc']
		if hit_dist != 0:
			c = hit_dist / math.sqrt(hit_coord[0]**2+hit_coord[1]**2) #adjust hit_coord using hit_dist (assume hit_dist is more accurate)
			if not np.isnan(c):
				hit_coord = (hit_coord[0]*c, hit_coord[1]*c)
		batter = {'rhb':row['stand'], 't_first':4} #TODO personalize t_first to batter
		if not ignore_foul or hit_coord[1] > abs(hit_coord[0]):
			hits.append({'launch_angle':launch_angle, 'exit_velocity':exit_velo, 'hangtime':hangtime, 
				'height':height, 'hit_coord':hit_coord, 'weight':row['weight'], 'batter':batter})
	if output_file != None:
		with open(output_file, 'w') as f:
			hitstr = [str(x) for x in hits]
			f.write('\n'.join(hitstr))
	return hits

def evaluate_on_hit(hit):
	closest = list(range(1,10))

	launch_angle = hit['launch_angle']
	exit_velocity = hit['exit_velocity']
	hit_coord = hit['hit_coord']
	batter = hit['batter']
	weight = hit['weight']
	hangtime = hit['hangtime']
	h_lambda = hit['height']
	value = 1.0

	hc_x, hc_y = hit_coord
	dir_rad = tf.math.atan(hc_x/hc_y)
	la_rad = launch_angle*math.pi/180
	hit_slope = hc_y/hc_x
	ev_vert = exit_velocity*tf.math.sin(la_rad)
	ev_horz = exit_velocity*tf.math.cos(la_rad)
	for player in fielders:
		v = 5280/3600*ev_horz # ft/s
		s = 5280/3600*player['speed'] # ft/s
		#use quadratic formula to find fastest time to ball
		vx = v * tf.math.sin(dir_rad)
		vy = v * tf.math.cos(dir_rad)
		a = v**2 - s**2
		b = -2*player['loc'][0]*vx - 2*player['loc'][1]*vy
		c = tf.norm(player['loc'])**2
		disc = b**2-4*a*c #discriminant
		if disc < 0:
			continue
		t = (-b - tf.math.sqrt(disc)) / (2*a)
		t += player['reaction_time']
		#get trajectory info
		t_height = h_lambda(t)
		#if negative, groundball
		if  t>0 and t_height < 0:
			dist_home = v*t
			dist_first = tf.math.sqrt(90**2+dist_home**2-2*90*dist_home*tf.math.cos(math.pi/4-dir_rad)) #law of cosines
			t_gather = 0.5
			t_throw = dist_first/(5280/3600*player['throw_speed'])
			value_a = sinemoid(center=batter['t_first'], horz_compress=.5)(t+t_gather+t_throw)
			if value_a < value:
				closest = [player['pos']]
			if value_a == value:
				closest.append(player['pos'])
			value = tf.math.minimum(value, value_a)
		#flyball/line drive
		else:
			#line drive
			value_b = sinemoid(center=player['v_reach'], horz_compress=.5)(t_height) #TODO when t_height=32.3628 and v_reach=8, value_a=0.512179
			if value_b < value:
				closest = [player['pos']]
			if value_b == value:
				closest.append(player['pos'])
			value = tf.math.minimum(value, value_b)
			#fly ball
			time_to_spot = player['reaction_time'] + tf.norm(player['loc']-[hc_x, hc_y]) / s
			value_c = sinemoid(center=hangtime, horz_compress=.5)(time_to_spot)
			if value_c < value:
				closest = [player['pos']]
			if value_c == value:
				closest.append(player['pos'])
			value = tf.math.minimum(value, value_c)
		value = tf.math.maximum(value, 1-sinemoid(center=player['reaction_time'], horz_compress=.5)(t)) #reaction time
		if value == 1:
			closest = [np.argmin([tf.norm(player['loc']-hit_coord) for player in fielders])+1]
	return value, list(set(closest))

def evaluate_on_hit_discrete(hit):
	launch_angle = hit['launch_angle']
	exit_velocity = hit['exit_velocity']
	hit_coord = hit['hit_coord']
	batter = hit['batter']
	weight = hit['weight']
	hangtime = hit['hangtime']
	h_lambda = hit['height']
	value = 1.0

	hc_x, hc_y = hit_coord
	dir_rad = tf.math.atan(hc_x/hc_y)
	la_rad = launch_angle*math.pi/180
	hit_slope = hc_y/hc_x
	ev_vert = exit_velocity*tf.math.sin(la_rad)
	ev_horz = exit_velocity*tf.math.cos(la_rad)
	for player in fielders:
		v = 5280/3600*ev_horz # ft/s
		s = 5280/3600*player['speed'] # ft/s
		#use quadratic formula to find fastest time to ball
		vx = v * tf.math.sin(dir_rad)
		vy = v * tf.math.cos(dir_rad)
		a = v**2 - s**2
		b = -2*player['loc'][0]*vx - 2*player['loc'][1]*vy
		c = tf.norm(player['loc'])**2
		disc = b**2-4*a*c #discriminant
		if disc < 0:
			continue
		t = (-b - tf.math.sqrt(disc)) / (2*a)
		t += player['reaction_time']
		#get trajectory info
		t_height = h_lambda(t)
		#if negative, groundball
		if  t>0 and t_height < 0:
			dist_home = v*t
			dist_first = tf.math.sqrt(90**2+dist_home**2-2*90*dist_home*tf.math.cos(math.pi/4-dir_rad)) #law of cosines
			t_gather = 0.5
			t_throw = dist_first/(5280/3600*player['throw_speed'])
			if batter['t_first'] > t+t_gather+t_throw:
				return 0
		#flyball/line drive
		else:
			#line drive
			if player['v_reach'] > t_height:
				return 0
			#fly ball
			time_to_spot = player['reaction_time'] + tf.norm(player['loc']-[hc_x, hc_y]) / s
			if hangtime > time_to_spot:
				return 0
	return 1

def evaluate_alignment(batted_balls):
	value = 0
	for ball in batted_balls:
		value += (evaluate_on_hit_discrete(ball) * ball['weight'])
	return value / len(batted_balls)


def gradient_descent(batted_balls, epochs=10, if_lr=1e4, of_lr=1e4, decay=0, display=False): #TODO verbose setting
	print('Starting gradient descent.')
	for epoch in range(epochs):
		ball_dots = []
		ball_colors = []
		ball_sizes = []
		print('Epoch', epoch)
		with tf.GradientTape(persistent=True) as tape:
			woba = tf.Variable(0., dtype=float)
			tape.watch(woba)
			for player in fielders:
				tape.watch(player['loc'])
			for ball in batted_balls:
				value, closest = evaluate_on_hit(ball)
				if display:
					ball_dots.append(ball['hit_coord'])
					ball_sizes.append(ball['weight']*2)
					if float(value)<0.5:
						ball_colors.append('red') #RED = OUT
					else:
						ball_colors.append('green') #GREEN = HIT
				woba = woba + value*ball['weight']
			woba = woba / len(batted_balls)
		if display:
			player_dots = [tuple(player['loc'].numpy()) for player in fielders]
			print('BABIP:', len([c for c in ball_colors if c=='green'])/len(ball_colors))
			draw_field(330, 420, players=player_dots, balls=ball_dots, ball_colors=ball_colors, ball_sizes=ball_sizes)
		for player in fielders[2:]:
			gradient = tape.gradient(woba, player['loc'])
			if not gradient==None:
				lr = if_lr if player['pos']<7 else of_lr
				print('ADJUSTMENT (position {0}):'.format(player['pos']), gradient.numpy()*lr)
				new_loc = player['loc'] - gradient*lr
				# if float(player['pos']) != 3 or float(tf.norm(new_loc-[90/2**0.5,90/2**0.5])) <= player['max_dist']:
				player['loc'] = new_loc
				fix_1b()
				if player['loc'][0] > player['loc'][1]:
					player['loc'] = tf.Variable([abs(player['loc'][0]+player['loc'][1])/2, (player['loc'][0]+player['loc'][1])/2])
		if_lr *= 1 / (1 + decay)
		of_lr *= 1 / (1 + decay)
		print('WOBA value:',float(woba))

def centroid_adjust(batted_balls, epochs=1, weight=0.9):
	player_balls = [[] for _ in range(9)]
	print('Starting centroid adjustment.')
	for epoch in range(epochs):
		print('Epoch', epoch)
		for ball in batted_balls:
			value, closest = evaluate_on_hit(ball)
			if len(closest) == 1:
				for player in closest:
					player_balls[player-1].append(np.array(ball['hit_coord']))
		for player, balls in zip(fielders[2:], player_balls[2:]):
			if balls:
				player_dist = tf.norm(player['loc'])
				new_loc = tf.Variable(sum(balls)/len(balls), dtype=float)
				# if float(player['pos']) != 3 or float(tf.norm(new_loc-[90/2**0.5,90/2**0.5])) <= player['max_dist']:
				player['loc'] = new_loc*weight+player['loc']*(1-weight)
				fix_1b()
				if player['loc'][0] > player['loc'][1]:
					player['loc'] = tf.Variable([abs(player['loc'][0]+player['loc'][1])/2, (player['loc'][0]+player['loc'][1])/2])
				new_dist = tf.norm(player['loc'])
				if new_dist < player_dist: #if player moved in, move them backwards
					player['loc'] = player['loc']*player_dist/new_dist

def fix_1b():
	player = fielders[2]
	dist_vector = player['loc']-[90/2**0.5,90/2**0.5]
	dist = tf.norm(dist_vector)
	if dist > player['max_dist']:
		player['loc'] = dist_vector*player['max_dist']/dist + [90/2**0.5,90/2**0.5]

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
	# print('Starting gradient descent for', name[1], name[0])

	standard_value = evaluate_alignment(data)
	for _ in range(5):
		centroid_adjust(data, epochs=10, weight=1/2)
		gradient_descent(data, epochs=5, if_lr=1e4, of_lr=1e4, display=False)
	gradient_descent(data, epochs=15, if_lr=1e4, of_lr=1e4, display=False)
	player_dots = [tuple(player['loc'].numpy()) for player in fielders]
	shifted_value = evaluate_alignment(data)
	print(player_dots)
	print('Standard Value', standard_value)
	print('Shifed Value', shifted_value)
	draw_field(330, 420, players=player_dots, balls=hit_dots,title=f'Generated shift for {name[1]} {name[0]}'.upper())



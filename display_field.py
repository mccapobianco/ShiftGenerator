import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Rectangle
from numpy import sqrt, arcsin, pi

def draw_field(line_dist, center_dist, players=[], balls=[], ball_colors=[], ball_sizes=[], title=''):
	if balls and not ball_colors:
		ball_colors = ['green']*len(balls)
	if balls and not ball_sizes:
		ball_sizes = [2]*len(balls)
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	arc_center = (line_dist**2-center_dist**2)/(line_dist*sqrt(2)-2*center_dist)
	radius = center_dist - arc_center
	theta = arcsin(line_dist*sqrt(2)/(2*radius))*180/pi
	if line_dist > center_dist:
		theta = -theta
	height = center_dist-line_dist/sqrt(2)
	max_dist = max(line_dist, center_dist)
	plt.plot([0,line_dist/sqrt(2)],[0,line_dist/sqrt(2)], color="black")
	plt.plot([0,-line_dist/sqrt(2)], [0,line_dist/sqrt(2)], color="black")
	plt.plot([-line_dist/sqrt(2),line_dist/sqrt(2)], [max_dist+10,max_dist+10], color="white")	
	base_size = 3
	#first base
	plt.plot([90/sqrt(2), 90/sqrt(2)-base_size],[90/sqrt(2), 90/sqrt(2)+base_size], color='black')
	plt.plot([90/sqrt(2), 90/sqrt(2)-base_size],[90/sqrt(2)+base_size*2, 90/sqrt(2)+base_size], color='black')
	plt.plot([90/sqrt(2), 90/sqrt(2)+base_size],[90/sqrt(2)+base_size*2, 90/sqrt(2)+base_size], color='black')
	#second base
	plt.plot([0, -base_size],[90*sqrt(2), 90*sqrt(2)+base_size], color='black')
	plt.plot([0, -base_size],[90*sqrt(2)+base_size*2, 90*sqrt(2)+base_size], color='black')
	plt.plot([0, base_size],[90*sqrt(2)+base_size*2, 90*sqrt(2)+base_size], color='black')
	plt.plot([0, base_size],[90*sqrt(2), 90*sqrt(2)+base_size], color='black')
	#third base
	plt.plot([-90/sqrt(2), -(90/sqrt(2)-base_size)],[90/sqrt(2), 90/sqrt(2)+base_size], color='black')
	plt.plot([-90/sqrt(2), -(90/sqrt(2)-base_size)],[90/sqrt(2)+base_size*2, 90/sqrt(2)+base_size], color='black')
	plt.plot([-90/sqrt(2), -(90/sqrt(2)+base_size)],[90/sqrt(2)+base_size*2, 90/sqrt(2)+base_size], color='black')

	fence = Arc((0, arc_center),height=2*radius,width=2*radius,angle=0, theta1=90-theta, theta2=90+theta, color="black")
	ax.add_patch(fence)
	infield = Arc((0, 0),height=320,width=150*sqrt(2),angle=0, theta1=45, theta2=135, color="black")
	ax.add_patch(infield)
	mound = plt.Circle((0, 60.6), 10, color='black', fill=False)
	ax.add_patch(mound)
	for player in players:
		pos = plt.Circle(player, 5, color='blue')
		ax.add_patch(pos)
	for ball, color, size in zip(balls, ball_colors, ball_sizes):
		pos = plt.Circle(ball, size, color=color)
		ax.add_patch(pos)
	plt.title(title)
	plt.show()

if __name__ == '__main__':
	import sys
	draw_field(int(sys.argv[1]), int(sys.argv[2]))
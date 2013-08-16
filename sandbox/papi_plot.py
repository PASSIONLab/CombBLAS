#!/usr/local/bin/python
import sys, re, math
from matplotlib import pyplot as plt
import numpy
import matplotlib

import os, datetime

input_files = sorted(sys.argv[1:])

np = len(input_files)
events = None 
labels = []
max_iter = 0
data = []
time_event = None

scatter_dump_out_filename = "scatter_dump_" + sys.argv[-1].replace(".txt", ".png")
time_correlation_scatter_out_filename = "time_correlation_scatter_" + sys.argv[-1].replace(".txt", ".png")

def modification_date(filename):
	try:
		t = os.path.getmtime(filename)
		return datetime.datetime.fromtimestamp(t)
	except OSError:
		return datetime.datetime(datetime.MINYEAR, 1, 1)

# figure out if any inputs have changed since the last run
def update_needed():
	newest_infile_time = modification_date(sys.argv[0]) # modification time of this script
	for f in input_files:
		newest_infile_time = max(newest_infile_time, modification_date(f))

	oldest_out = min(modification_date(scatter_dump_out_filename),
					modification_date(time_correlation_scatter_out_filename))
	
	return oldest_out < newest_infile_time

if not update_needed():
	print "Outputs up to date. Quitting."
	sys.exit()

#sys.exit()

# pass 1 - load
p = 0
for filename in input_files:
	with open(filename, "r") as f:
		for line in f:
			if line.find("iter") != -1: # header line
				if not events:
					h = line.strip().split("\t")
					events = h[2:]
			else:
				d = line.strip().split("\t")
				iter = int(d[0])
				label = d[1]
				
				if label not in labels:
					labels.append(label)
				max_iter = max(iter, max_iter)
				dp = {'p':p, 'iter':iter, 'label':label}
				
				if len(events) != len(d[2:]):
					raise ValueError, "length mismatch!"
					
				for x in zip(events, d[2:]):
					dp[x[0]] = x[1]
				data.append(dp)
	p += 1				

time_event_i = 0
for e in events:
	if e.find("TIME") != -1:
		time_event = e
		break
	time_event_i += 1

print "Events:", events
print "TIME event: ", time_event, "idx=", time_event_i
print "Labels:", labels
print "max_iter:", max_iter
print "len(data): ", len(data)
#for d in data:
#	print d

# Figure out the plot title
title = sys.argv[-1]
#print title
title = title.replace("temp_papi_output_", "")
#print title
title = re.sub("_p...txt", "", title)
#print title
title = re.sub("_events[A-Z0-9\-_]*", "_", title)
print "Title: ", title

summarize = True if title.find("start") != -1 else False

markers = ['x', 'o', 's', '*', '+', 'd']

# color codes from http://www.tayloredmktg.com/rgb/
blues = ["#000080", "#0000cd", "#1e90ff", "#00bfff", "#87ceeb", "#add8e6"]
greens = ["#006400", "#2e8b57", "#3cb371", "#8fbc8f", "#00ff7f", "#98fb98"]
yellows = ["#b8860b", "#daa520", "#ffd700", "#eedd82", "#ffff00", "#eee8aa"]
browns = ["#8b4513", "#a52a2a", "#a0522d", "#cd853f", "#f4a460", "#deb887"]
pinks = ["#b03060", "#d02090", "#db7093", "#ff1493", "#ff69b4", "#ffb6c1"]
violets = ["#8a2be2", "#a020f0", "#ba55d3", "#ee82ee", "#dda0dd", "#d8bfd8"]

if np == 1:
	colors = ['r']
if np == 4:
	colors = ['r', 'g', 'b', 'y']
if np == 9:
	colors = blues[0:3] + greens[0:3] + yellows[0:3]
elif np == 16:
	colors = blues[0:4] + greens[0:4] + yellows[0:4] + violets[0:4]
elif np == 25:
	colors = blues[0:5] + greens[0:5] + yellows[0:5] + pinks[0:5] + violets[0:5]
elif np == 36:
	colors = blues + greens + yellows + browns + pinks + violets



############################
# Report: scatter plot of all events vs. iteration #
def scatter_dump():
	global data, events, labels, max_iter, np
	
	fig, axarr = plt.subplots(len(events), sharex=True)

	for event in events:
		plot_id = events.index(event)
		
		sx = []
		sy = []
		sc = []
		for dp in data:
			x = dp['iter']*len(labels) + labels.index(dp['label'])
			sx.append(x)
			sy.append(dp[event])
			sc.append(colors[dp['p']]) # color
	
		axarr[plot_id].scatter(sx, sy, c=sc, marker='x', label=None)
		axarr[plot_id].set_ylabel(event)
		axarr[plot_id].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
		#plt.legend(loc=2)

	# set up the ticks
	plt.xlabel("counters")
	ax = plt.gca()
	num_ticks = len(labels)*(max_iter+1)
	ax.set_xticks(range(num_ticks))
	ticklabels = []
	for i in range(num_ticks):
		l = labels[i%len(labels)] + " %d"%(int(i/(len(labels))))
		ticklabels.append(l)
	ax.set_xticklabels(ticklabels)
	plt.setp(ax.get_xticklabels(), rotation='vertical')

	plt.xlim([-1,num_ticks+1])
	
	fig.suptitle(title, fontsize=18)

	#define plot size in inches (width, height) & resolution(DPI)
	fig.set_size_inches(18.5,20.5)
	plt.tight_layout()
	plt.subplots_adjust(top=0.93)

	print("saving figure to " + scatter_dump_out_filename)
	plt.savefig(scatter_dump_out_filename, dpi=200)

############################
# Report: scatter plot of all events vs. time

def summarize_data(data):
	import sys
	
	def get_key(dp):
		return (dp['iter'], dp['label'])#, dp['p'])
	def undo_get_key(k):
		return {'iter': k[0], 'label': k[1]}
	
	# summarize all 
	data_dict = {}
	for dp in data:
		k = get_key(dp)
		if k not in data_dict:
			dp_dict = {}
			for e in events:
				dp_dict[e] = []
			data_dict[k] = dp_dict
		else:
			dp_dict = data_dict[k]

		for e in events:
			dp_dict[e].append(float(dp[e]))
	
	s_data = []
	for k, dpd in data_dict.items():
		sdp = undo_get_key(k)
		
		for e in events:
			sdp["%s min"%e] = numpy.min(dpd[e])
			sdp["%s max"%e] = numpy.max(dpd[e])
			sdp["%s median"%e] = numpy.median(dpd[e])
		
		s_data.append(sdp)
	
	#for sdp in s_data:
	#	print sdp
	return s_data

if summarize:
	summarized_data = summarize_data(data)

def time_correlation_scatter():
	global data, events, labels, max_iter, np, time_event, time_event_i
	
	if time_event is None:
		print "time_correlation_scatter(): No time event to correlate with. Stopping"
		return
	
	if len(events) == 5:
		fig, axarr_4 = plt.subplots(2, 2)
		axarr = []
		for a in axarr_4:
			for b in a:
				axarr.append(b)
	else:
		fig, axarr = plt.subplots(len(events), sharex=True)
	#axarr[0].plot(x, y)
	#axarr[0].set_title('Sharing X axis')
	#axarr[1].scatter(x, y)

	plot_id = 0
	for event in events:
		if event == time_event:
			continue
		
		# basic plot
		label_groups = {}
		sx = []
		sy = []
		sc = []
		mk = []
		for dp in data:
			L = dp['label']
			if L not in label_groups:
				label_groups[L] = {}
				label_groups[L]['sx'] = []
				label_groups[L]['sy'] = []
				label_groups[L]['sc'] = []
				label_groups[L]['marker'] = markers[labels.index(L)]

			label_groups[L]['sx'].append(dp[time_event])
			label_groups[L]['sy'].append(dp[event])
			label_groups[L]['sc'].append(colors[dp['p']]) # color
		
		legend_p = []
		legend_l = []
		for L, Lg in label_groups.items():
			p = axarr[plot_id].scatter(Lg['sx'], Lg['sy'], c=Lg['sc'], marker=Lg['marker'], lw=0.5)
			legend_p.append(p)
			legend_l.append(L)
		
		# error bars
		if summarize:
			x = []
			y = []
			time_err_min = []
			time_err_max = []
			e_err_min = []
			e_err_max = []
			for sdp in summarized_data:
				x.append(sdp['%s median'%time_event])
				y.append(sdp['%s median'%event])

				time_err_min.append(x[-1] - sdp['%s min'%time_event])
				time_err_max.append(sdp['%s max'%time_event] - x[-1])

				e_err_min.append(y[-1] - sdp['%s min'%event])
				e_err_max.append(sdp['%s max'%event] - y[-1])
				
				axarr[plot_id].annotate("%s %d"%(sdp['label'], sdp['iter']), xy=(x[-1], y[-1]), xytext=(-28,20), 
					textcoords='offset points', ha='center', va='bottom',
					bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
					arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', 
									color='red'))
                            
			axarr[plot_id].errorbar(x, y, xerr=[time_err_min, time_err_max], yerr=[e_err_min, e_err_max], ecolor='k', fmt=None)
			
		# legends
		if plot_id != 0:
			reg_legend = axarr[plot_id].legend(legend_p, legend_l,
			   scatterpoints=1,
			   loc='upper left',
			   ncol=4,
			   fontsize=10)
		else:
			color_p = []
			color_l = []
			for ci in range(len(colors)):
				color_p.append( plt.Rectangle((0, 0), 1, 1, fc=colors[ci]) )
				color_l.append("p%2d"%ci)
			color_legend = axarr[plot_id].legend(color_p, color_l, scatterpoints=1, loc='upper left', ncol=int(math.sqrt(len(colors))), fontsize=10, columnspacing=0.7)
			#axarr[plot_id].add_artist(reg_legend)

		axarr[plot_id].set_ylabel(event)
		# set up the ticks
		axarr[plot_id].set_xlabel(time_event)
		
		# scientific notation at all times
		axarr[plot_id].ticklabel_format(style='sci', axis='both', scilimits=(0,0))
		
		plot_id += 1		

	fig.suptitle(title, fontsize=18)
	
	
	
	#define plot size in inches (width, height) & resolution(DPI)
	fig.set_size_inches(15, 15)
	plt.tight_layout()
	plt.subplots_adjust(top=0.93)

	print("saving figure to " + time_correlation_scatter_out_filename)
	plt.savefig(time_correlation_scatter_out_filename, dpi=200)

scatter_dump()
time_correlation_scatter()

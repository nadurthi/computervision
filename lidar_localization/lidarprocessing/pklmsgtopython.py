import pickle

with open("houseScan.pkl",'rb') as fh:
	ff=pickle.load(fh)

L=[]
i=0
for msg in ff:
	L.append({'ranges':msg.ranges,
		'intensities':msg.intensities,
		'range_max':msg.range_max,
		'range_min':msg.range_min,
		'scan_time':msg.scan_time,
		'time_increment':msg.time_increment,
		'angle_increment':msg.angle_increment,
		'angle_max':msg.angle_max,
		'angle_min':msg.angle_min,

		})
	i+=1
	print(i)

with open("houseScan_std.pkl",'wb') as fh:
	pickle.dump(L,fh)



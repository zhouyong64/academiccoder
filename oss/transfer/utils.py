# -*- coding: utf-8 -*-
import os
import numpy as np
import codecs

DELI = ' @#@# '
DELIMIT = ' @#@# '

def overlap(sa,sb):
	all = set(sa)
	if all.isdisjoint(set(sb)):
		return np.inf
	for i in sb:
		all.add(i)
	lap = set(sa).intersection(all)
	#cost = np.exp(-1.0*len(lap)/len(all))
	cost = -1.0*len(lap)/len(all)
	return cost

def dish_dist(da,db,allowPOS=[]):
# 	print('da',da)
# 	print(db)
	dac = [w[0] for w in da if len(w) > 1 and w[1] in allowPOS]
	dbc = [w[0] for w in db if len(w) > 1 and w[1] in allowPOS]
	return overlap(dac,dbc)

def get_lines(fname,split=False):
        lines = []
        segs = []
        with codecs.open(fname, "r", "utf-8") as fr:
                for line in fr.readlines():
                        line = line.strip()
                        if len(line) > 0:
                                if split:
                                        lines.append(line)
                                        segs.append(line.split(' '))
                                else:
                                        lines.append(line)
        return np.array(lines),segs

def get_lines2(fname,split=False):
        lines = []
        segs = []
        with codecs.open(fname, "r", "utf-8") as fr:
                for line in fr.readlines():
                        line = line.strip()
                        ss = line.split(DELIMIT)
                        if len(ss) > 1:
                                if split:
                                        lines.append(ss[1])
                                        segs.append(ss[0].split(' '))
                                else:
                                        lines.append(ss[1])
        return np.array(lines),segs	

def get_lines3(fname,split=False):
	lines = []
	segs = []
	with codecs.open(fname, "r", "utf-8") as fr:
		for line in fr.readlines():
			line = line.strip()
			if len(line) > 0:
				ss = line.split(DELIMIT)
				if len(ss) > 1:
					if split:
						lines.append(ss[1])
						splits = ss[0].split('  ')
						sss = [s.split(' ') for s in splits]
						segs.append(sss)
					else:
						lines.append(line)
						
	return np.array(lines),np.array(segs)


def temp():
	s=u'蚂蚁 上树食 材包 @#@# 加粉'
	ss = s.split(DELI)
	print(ss[0],ss[1])
	cost = overlap(ss[0].split(' '),ss[1].split(' '))
	print('cost',cost)


if __name__ == '__main__':
	temp()

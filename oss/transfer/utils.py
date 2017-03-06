# -*- coding: utf-8 -*-
import os
import numpy as np

DELI = ' @#@# '

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
	
def temp():
	s=u'蚂蚁 上树食 材包 @#@# 加粉'
	ss = s.split(DELI)
	print ss[0],ss[1]
	cost = overlap(ss[0].split(' '),ss[1].split(' '))
	print 'cost',cost


if __name__ == '__main__':
	temp()

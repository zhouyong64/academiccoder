# -*- coding: utf-8 -*-
import os
import numpy as np
import codecs
import editdistance as edist
import urllib.request
import urllib.parse
import time

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


def get_lines4(fname,split=False):
        lines = []
        segs = []
        with codecs.open(fname, "r", "utf-8") as fr:
                for line in fr.readlines():
                        line = line.strip()
                        ss = line.split(DELIMIT)
                        if len(ss) > 1:
                                if split:
                                        lines.append(ss[0])
                                        segs.append(ss[1].split(' '))
                                else:
                                        lines.append(ss[0])
        return np.array(lines),segs	

def temp():
	s=u'蚂蚁 上树食 材包 @#@# 加粉'
	ss = s.split(DELI)
	print(ss[0],ss[1])
	cost = overlap(ss[0].split(' '),ss[1].split(' '))
	print('cost',cost)
	
def random_select(srcf,outf,num=1100):
	lines = []
	with codecs.open(srcf, "r", "utf-8") as fr:
		for line in fr.readlines():
			line = line.strip()
			lines.append(line)
	idxs = np.random.permutation(len(lines))[:num*10]
	out = codecs.open(outf, "w", "utf-8")
	c =0
	for i,idx in enumerate(idxs):
		if c>num:
			break
		if lines[idx].count('+') < 1 and lines[idx].count('＋') < 1 and lines[idx].count('➕') < 1:
			txt = lines[idx].replace(' ','')
			out.write(txt + '\n')
			c+=1
	out.close()

def get_voc(vocf):
	voc = {}
	with codecs.open(vocf, "r", "utf-8") as fr:
		for line in fr.readlines():
			line = line.strip()
			ss = line.split(' ')
			if len(ss) < 2:
				voc[ss[0]] = ''
				continue
			voc[ss[1]] = ss[0]
	return voc
			
def idx2names(srcf,dstf,vocf):
	voc = get_voc(vocf)
	with codecs.open(dstf, "w", "utf-8") as fw:
		with codecs.open(srcf, "r", "utf-8") as fr:
			for line in fr.readlines():
				line = line.strip()
				ss = line.split(DELI)
				if len(ss) < 3:
					continue
				raw = ss[0].split(' ')
				weak_lab= ss[1].split(' ')
				pred = ss[2].split(' ')
				
				raw = [voc[idx] for idx in raw]
				weak_lab = [voc[idx] for idx in weak_lab]
				pred = [voc[idx] for idx in pred]
				
				raw = ''.join(raw)
				weak_lab = ''.join(weak_lab)
				pred = ''.join(pred)
				
				fw.write(raw + DELI + weak_lab + DELI + pred + '\n')

def filter_voc(src, target, dstf):
	_,segs = get_lines(src, split=True)
	done = [i[0] for i in segs]
	
	_,segs = get_lines(target, split=True)
	candi = [i for i in segs if i[0] not in done]
	
	with codecs.open(dstf, "w", "utf-8") as fw:
		for word in candi:
			fw.write(' '.join(word) + '\n')

'''
src is clustering result txt, using edit distance to see close clusters
'''
def analyze_cluster(src, dstf):
	_,segs = get_lines(src, split=True)
	clusters  = [i[0] for i in segs]
	num = len(clusters)
	dists = np.ones((num,num))*-1
	with codecs.open(dstf, "w", "utf-8") as fw:
		for i in range(num):
			fw.write(clusters[i] + ' ')
			for j in range(num):
				if i == j or dists[i,j] != -1:
					continue
				dist = edist.eval(clusters[i],clusters[j])
				dists[i,j] = dist
				if dist == 1:
					fw.write(clusters[j] + ' ')
			fw.write('\n')
	
	d0_mask = (dists==0).nonzero()
	d1_mask = (dists==1).nonzero()
	d2_mask = (dists==2).nonzero()
	d0 = len(d0_mask[0])
	d1 = len(d1_mask[0])
	d2 = len(d2_mask[0])
	print('dist summary',d0,d1,d2)
# 	with codecs.open(dstf, "w", "utf-8") as fw:
# 		fw.write('dist 1\n')
# 		for i in range(len(d1_mask[0])):
# 			ii = d1_mask[0][i]
# 			jj = d1_mask[1][i]
# 			fw.write(' '.join(word) + '\n')

def extract_voc_from_txt(srcf,dstf,vocf):
	_,segs = get_lines(vocf,split=True)
	voc = [s[0] for s in segs]
	newv = set()
	with codecs.open(dstf, "w", "utf-8") as fw:
		with codecs.open(srcf, "r", "utf-8") as fr:
			for line in fr.readlines():
				line = line.strip()
				if len(line) < 1:
					continue
				ss = line.split(' ')
				for seg in ss:
					if seg not in voc:
						newv.add(seg)
		for v in newv:
			fw.write(v + '\n')

def testHttp(vocf):
	cands,_ = get_lines(vocf,split=False)
	for i,cand in enumerate(cands):
		stime = np.random.randint(10,30)
		time.sleep(stime)
		params = urllib.parse.urlencode({'q': cand})
		url = "http://www.xinshipu.com/doSearch.html?%s" % params
		with urllib.request.urlopen(url) as f:
			dstf = 'spider/' + cand + '.html'
			with codecs.open(dstf, "w", "utf-8") as fw:
				fw.write(f.read().decode('utf-8'))
											
if __name__ == '__main__':
# 	temp()
	base = '/Users/joe/Downloads/dish/'
# 	idx2names(base + 'preds.txt',base + 'preds_names.txt',base + 'embed_data_norm_voc.txt')
# 	filter_voc(base + 'duang_dish_raw_voc_n_100t_fix.txt',base + 'duang_dish_raw_voc_n_50t.txt',\
# 			base + 'duang_dish_raw_voc_n_50t_candi.txt')
# 	analyze_cluster(base + 'cluster10k.txt',base + 'cluster10k_d1.txt')
# 	extract_voc_from_txt(base + 'online.txt',base + 'voc_from_online.txt',\
#  			base + 'duang_dish_raw_voc_n_100t_fix.txt')
	testHttp('voc_candi.txt')
# 	random_select('/Users/joe/Downloads/dish/duang_dish_tail490w_raw_cleanByRE2.txt','/Users/joe/Downloads/dish/testset1k.txt')

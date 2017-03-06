# -*- coding: utf-8 -*-
import re
import codecs
import numpy as np
from utils import overlap

DELIMIT = ' @#@# '

def printable(ss):
	if type(ss) == type([]):
		ret = [unicode(s).encode('utf8') for s in ss]
		return '\n'.join(ret)
	else:
		return unicode(ss).encode('utf8')

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

def remove_emoji(fname,dst):
	emoji_pattern = re.compile(
    		u"(\ud83d[\ude00-\ude4f])|"  # emoticons
    		u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
    		u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
    		u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
    		u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
    		"+", flags=re.UNICODE)
	with codecs.open(fname, "r", "utf-8") as fr:
        	with codecs.open(dst, "w", "utf-8") as fw:
            		for line in fr.readlines():
                		line = line.strip()
				removed = emoji_pattern.sub(r'', line)
				fw.write(removed + '\n')

'''
Given seg result "火 龙果", if there is an entry "火龙果" in the dictionary, we can detect this seg error.
'''
def get_seg_wrong_candidates():
	segf = '/Users/joe/Downloads/dish1m_cleanByRE_seg.txt'
	dictf = '/Users/joe/Downloads/webdict-master/webdict_with_freq.txt'
	outf = '/Users/joe/Downloads/seg_wrong_candidates.txt'
	_,voclist = get_lines(dictf,split=True)
	voc = [v[0] for v in voclist]
	print('len of dict: ',len(voc))
	print(printable(voc[-100]))

	_,segs = get_lines(segf,split=True)
	
def segment(srcf,dstf):
	import jieba
	lines,_ = get_lines(srcf,split=False)
	with codecs.open(dstf, "w", "utf-8") as fw:
		for line in lines:
			print(line)
			segs = jieba.cut(line)
			print("Full Mode: " + "/ ".join(segs))
			fw.write(u"烂东西\n")
			fw.write(' '.join(segs) + '\n')
			break

def generate_data(outf,nseed=10,top=20,each=5):
        from nltk.metrics.distance import edit_distance
        from dtw import dtw

        dist_fun = edit_distance
        df = '/Users/joe/Downloads/dish1m_cleanByRE2_segByJieba.txt'
        whole_d,dishes = get_lines2(df,split=True)

	out = codecs.open(outf, "w", "utf-8")
        for i in range(nseed):
		if (i+1) % 500 == 0:
			print('i',i)
                idx = np.random.randint(1,len(dishes))
                dish = dishes[idx]
                min_dist = np.array([np.inf]*top)
                min_idx = np.array([-1]*top)

		max_dist = np.array([-np.inf]*top)
                max_idx = np.array([-1]*top)
                for j in range(len(dishes)):
                        if j == idx:
                                continue
                        if j % 100000 == 0:
                                #print('dtw processing',j)
                                pass
			if len(dishes[j])==1 and len(dishes[j][0]) == 1:
				continue
                        #dist, cost, acc = dtw(dish, dishes[j], dist_fun)
                        dist = overlap(dish,dishes[j])
			if dist != np.inf and dist < min_dist[-1]:
                                min_dist[-1] = dist
                                min_idx[-1] = j
                                min_idx = min_idx[np.argsort(min_dist)]
                                min_dist = np.sort(min_dist)

			if dist > max_dist[-1]:
                                max_dist[-1] = dist
                                max_idx[-1] = j
                                max_idx = max_idx[np.argsort(max_dist)]
                                max_dist = np.sort(max_dist)
		if (min_dist == np.inf).all():
			continue
		idxs = np.random.permutation(top)
		select = min_idx[idxs[:each]]
		
		for d in range(each):
			out.write('1 ' + whole_d[idx] + DELIMIT + whole_d[select[d]] + '\n')
			#out.write(' '.join(dishes[idx]) + DELIMIT + ' '.join(dishes[select[d]]) + '\n')
			#out.write('\n')
		

		'''
		if (max_dist == -np.inf).all():
                        continue
		'''
                idxs = np.random.permutation(len(dishes))
                #select = max_idx[idxs[:each]]
                select = idxs[:each]
                for d in range(each):
                        out.write('0 ' + whole_d[idx] + DELIMIT + whole_d[select[d]] + '\n')
                        #out.write(' '.join(dishes[idx]) + DELIMIT + ' '.join(dishes[select[d]]) + '\n')
                        #out.write('\n')
	out.close()




'''
find the most similar dishes by using dtw or simple "overlap" ratio
"segf" is the segmented result output by Java code
'''	
def find_similar(segf,top=10,ntry=1):
	from nltk.metrics.distance import edit_distance
	from dtw import dtw
	
	dist_fun = edit_distance
	whole_d,dishes = get_lines2(segf,split=True)

	for i in range(ntry):
		idx = np.random.randint(1,len(dishes))
		dish = dishes[idx]
		#dish = [u"奥利奥",u"草莓",u"夹心",u"饼干"]
		#dish = [u"上海滩",u"醉鱼"]
		#dish = [u"酸辣",u"土豆丝"]
		#dish = [u"倍思沃",u"原味",u"捞面"]
		#dish = [u"土豆",u"排骨饭"]
		#dish = [u"黑椒",u"牛肉",u"木桶饭"]
		#dish = [u"砂锅",u"烧鸡",u"面"]
		#dish = [u"砂锅",u"川味",u"馄饨"]
		min_dist = np.array([np.inf]*top)
		min_idx = np.array([-1]*top)
		for j in range(len(dishes)):
			if j == idx:
				continue
			if j % 100000 == 0:
				#print('dtw processing',j)
				pass
			#dist, cost, acc = dtw(dish, dishes[j], dist_fun)
			dist = overlap(dish,dishes[j])
			if dist < min_dist[-1]:
				min_dist[-1] = dist
				min_idx[-1] = j
				min_idx = min_idx[np.argsort(min_dist)]
				min_dist = np.sort(min_dist)
				'''
				print('j: ',j,dist)
				print('min dist',min_dist)
				print('min_idx',min_idx)
				'''
		print('most similar dish: ------------------------ ')
		print(printable(whole_d[idx]))
		print(printable(whole_d[min_idx].tolist()))
		print('\n')
		

if __name__ == '__main__':
	base = '/Users/joe/Downloads/'
	base = 'data/'
	fname = '/Users/joe/Downloads/dish5m.txt'
	dst = '/Users/joe/Downloads/dish5m_clean.txt'
	segf = base + '/dish1m_cleanByRE2_segByJieba.txt'
	#remove_emoji(fname,dst)
	find_similar(segf,top=30)
	#get_seg_wrong_candidates()
	#segment(base + 'dish1m_cleanByRE.txt', base + 'dish1m_cleanByRE2_segByJieba.txt')
	'''
	generate_data(base + 'training1.txt',nseed=5000,top=5,each=1)
	generate_data(base + 'training2.txt',nseed=5000,top=5,each=1)
	generate_data(base + 'training3.txt',nseed=5000,top=5,each=1)
	generate_data(base + 'training4.txt',nseed=5000,top=5,each=1)
	generate_data(base + 'training5.txt',nseed=5000,top=5,each=1)
	'''

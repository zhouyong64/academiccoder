import re
import codecs
import numpy as np

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
def segment(srcf,dstf):
        import jieba
        lines,_ = get_lines(srcf,split=False)
        with codecs.open(dstf, "w", "utf-8") as fw:
                for line in lines:
                        print(line)
                        segs = jieba.cut(line)
                        print("Full Mode: " + "/ ".join(segs))
                        fw.write(' '.join(segs) + '\n')
                        break


base = '/Users/joe/Downloads/'
segment(base + 'dish1m_cleanByRE.txt', base + 'dish1m_cleanByRE2_segByJieba.txt')

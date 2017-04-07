# -*- coding: utf-8 -*-
import random
import time
import urllib.request
import urllib.parse
import codecs
import os
from utils import extract_zones
from utils import get_lines

def http_request(start,end):
    dname = 'spider'
    if not os.path.exists(dname):
        os.mkdir(dname)
    for i in range(start,end):
        stime = random.randint(3,6)
        time.sleep(stime)
        url = "http://www.xinshipu.com/jiachangzuofa/%s/" % i
        print('processing',i,url)
        with urllib.request.urlopen(url) as f:
            dstf = dname + '/' + str(i) + '.html'
            with codecs.open(dstf, "w", "utf-8") as fw:
                fw.write(f.read().decode('utf-8'))
                

def http_request_for_site_url(dname,areas,pages_per_zone=3):
    base = 'http://www.lepu.cn/shop/'
    if not os.path.exists(dname):
        os.mkdir(dname)
#     for dis,zs in zip(distr,zones):
    for dis in areas.keys():
        dstd = dname + '/' + dis
        if not os.path.exists(dstd):
            os.mkdir(dstd)
        zs = areas[dis]
        for z in zs:
            for i in range(1,pages_per_zone+1):
                stime = random.randint(20,40)
                time.sleep(stime)
                url = base + dis + '/' + z + '/'
                if i > 1:
                    url += 'p' + str(i) + '/'
                print('processing',url)
                try:
                    with urllib.request.urlopen(url) as f:
                        dstf = dstd + '/' + z + str(i) + '.html'
                        with codecs.open(dstf, "w", "utf-8") as fw:
                            fw.write(f.read().decode('utf-8'))
                except:
                    pass

def http_request_for_site_data(urls,dname):
    if not os.path.exists(dname):
        os.makedirs(dname)
    for i,url in enumerate(urls):
        stime = random.randint(20,30)
        time.sleep(stime)
        print('processing',i,url)
        idn = url.split('/')[-2]
        try:
            with urllib.request.urlopen(url) as f:
                dstf = dname + '/' + idn + '.html'
                with codecs.open(dstf, "w", "utf-8") as fw:
                    fw.write(f.read().decode('utf-8'))
        except:
            pass
                                  
def crawl_site_url(save_dir,pages_per_zone=5):
    areas = {}
    extract_zones('sites_info/haidian_zones.txt',areas)
    extract_zones('sites_info/chaoyang_zones.txt',areas)
    http_request_for_site_url(save_dir,areas,pages_per_zone)

def crawl_site_data(save_dir,url_file):
    urls,_ = get_lines(url_file,split=False)
    http_request_for_site_data(urls,save_dir)
             
'''
把main中的your_range改成自己负责的区域，如(20001, 40000)，然后开始运行即可。
为了防止把服务器搞崩，也为了防止触动它可能的反爬虫机制，http_request中设置了不短的sleep time.

各自负责的range:
Zhouyong (0,20000)
Duang (20001, 40000)
Zhiqi (40001,60000)
Guangdong (60001,80000)
Juanchao (80001,98000)

'''
if __name__ == '__main__':
    start,end = (464,2000)
#     http_request(start,end)
#     http_request_for_site_data(['chaoyang','haidian'],[['wangjingdong','chaoyanggongyuan'],['wudaokou','xierqi']],2)
#     http_request_for_site_data(['haidian'],[['wudaokou','xierqi']],2)
#     crawl_site_url('spider_site/')
    crawl_site_data('sites_data/haidian/','sites_info/haidian_urls.txt')
    crawl_site_data('sites_data/chaoyang/','sites_info/chaoyang_urls.txt')
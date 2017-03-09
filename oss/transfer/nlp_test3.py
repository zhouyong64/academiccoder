import codecs
from utils import get_lines
from utils import get_lines3
from utils import dish_dist
from utils import DELIMIT
import numpy as np
import pickle
import jieba.posseg as pseg
import jieba
jieba.load_userdict("dish_dict.txt")


def get_normal_name(srcDish, sims, allowPOS = ['nr','nt','nz','n']):
    src = [w[0] for w in srcDish if len(w) > 1 and w[1] in allowPOS]
    nbs = []
    for sim in sims:
        segs = [w[0] for w in sim if len(w) > 1 and w[1] in allowPOS]
        nbs.extend(segs)
    norm = [seg for seg in src if nbs.count(seg) > 1]
    if len(norm) < 1:
        if len(src) > 0:
            return src
        else:
            return [w[0] for w in srcDish]
    return norm

def find_similar(segf,srcdishes=None,top=10,ntry=5,savef=None):
#         whole_d,dishes = get_lines3(segf,split=True)
        fd = open('dish1m_seg_originName.pkl','rb')
        whole_d,dishes = pickle.load(fd)
        fd.close()
        
        outf = None
        if savef is not None:
            outf = codecs.open(savef, "w", "utf-8")
        if srcdishes is not None:
            ntry = len(srcdishes)
        for i in range(ntry):
                if srcdishes is not None:
                    idx=0
                    dishName = srcdishes[i]
#                     dishName = '香干芹菜炒肉煲仔'
                    segs = pseg.cut(dishName)
                    dish = [[w,t] for w,t in segs]
                else:
                    idx = np.random.randint(1,len(dishes))
                    dish = dishes[idx]
                    dishName = whole_d[idx]

                min_dist = np.array([np.inf]*top)
                min_idx = np.array([-1]*top)
                for j in range(len(dishes)):
                    if j == idx:
                        continue
#                     if (j+1) % 10000 == 0:
#                         print('processing',j)
                    dist = dish_dist(dish,dishes[j],allowPOS = ['nr','nt','nz','n'])
                    if dist < min_dist[-1]:
                        min_dist[-1] = dist
                        min_idx[-1] = j
                        min_idx = min_idx[np.argsort(min_dist)]
                        min_dist = np.sort(min_dist)

                norm = get_normal_name(dish,dishes[min_idx])
                name = ''.join(norm)
                print('most similar dish: ------------------------ ')
                print(dishName)
                print('normalized name',name)
                print('\n'.join(whole_d[min_idx].tolist()))
                print('\n')
                if outf is not None:
                    msg = dishName + ' ' + name + ' ' + ','.join(whole_d[min_idx].tolist())
                    outf.write(msg + '\n')
        if outf is not None:
            outf.close()

                
def seg_and_save(srcf,outf):
        dishes,_ = get_lines(srcf)
        seglist = []
        with codecs.open(outf, "w", "utf-8") as fw:
                for i in range(len(dishes)):
                        dish = dishes[i]
                        words = pseg.cut(dish)
                        msg = ''
                        seg = []
                        for word, flag in words:
                                #print printable(word)
#                                 print('%s %s %d' % (word, flag, i))
                                msg += word + ' ' + flag + '  '
                                seg.append([word,flag])
                        seglist.append(seg)
                        msg = msg.strip()
                        msg += DELIMIT
                        msg += dish
                        fw.write(msg)
                        fw.write('\n')

        fd = open('dish1m_seg_originName.pkl','wb')
        pickle.dump((dishes,seglist),fd)
        fd.close()


if __name__ == '__main__':
    base = '/Users/joe/Downloads/dish/'
    fname = '/Users/joe/Downloads/dish5m.txt'
    dst = '/Users/joe/Downloads/dish5m_clean.txt'
    segf = base + '/dish1m_cleanByRE2_segByJieba.txt'
    unsegf = base + '/dish1m_cleanByRE.txt'
    #remove_emoji(fname,dst)
    #find_similar(segf,top=30)
#     seg_and_save(unsegf,'temp.txt')
    segf = 'tempDict_originName.txt'
#     find_similar(segf,top=20,ntry=100,savef='res.txt')
    dishes = ['香干芹菜炒肉煲仔','绍兴梁祝花雕','芒可椰果牛奶','浓汤牛腩拉面','黄焖鸡架','干锅阿婆土豆片','黄焖鸡盖浇饭汤','富贵肘子','香肠瘦肉饭','培根梅干菜寿司']
    dishes.extend(['阿克苏冰糖心苹果','酸菜肉丝酸辣粉','原汁鸡丝米线','白菜猪肉馅水饺','红烧茄子木桶饭','澳门葡式猪扒饭'])
    find_similar(segf,dishes,top=20,ntry=1,savef='res.txt')

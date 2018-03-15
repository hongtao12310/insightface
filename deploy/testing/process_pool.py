# coding: utf-8


from multiprocessing import Pool
from itertools import repeat
import time


def show_list(l):
    print(type(l))
    print(l)
    time.sleep(1)
    return l


if __name__ == '__main__':
    tbegin = time.time()
    l = [1, 2, 3, 4, 5]
    pool = Pool(processes=2)
    result = pool.map(show_list, repeat(l))
    print(result)
    tend = time.time()
    print("time eclipse: %s" %(tend-tbegin))


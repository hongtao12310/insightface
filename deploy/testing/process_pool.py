# coding: utf-8


from multiprocessing import Pool
from itertools import repeat
import time


def show_list(l):
    print(type(l))
    print(l)
    time.sleep(1)
    return l


def task(pid):
    time.sleep(1)
    # do something
    return pid


def main():
    pool = Pool(4)
    results = []
    for i in xrange(0, 4):
        result = pool.apply_async(task, args=(i,))
        results.append(result)

    print(results)
    print("close pool")
    pool.close()
    print("join pool")
    pool.join()
    print("after join")
    for result in results:
        print(result.get())


if __name__ == '__main__':
    # tbegin = time.time()
    # l = [1, 2, 3, 4, 5]
    # pool = Pool(processes=2)
    # result = pool.map(show_list, repeat(l))
    # print(result)
    # tend = time.time()
    # print("time eclipse: %s" %(tend-tbegin))
    main()


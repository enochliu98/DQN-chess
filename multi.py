from subprocess import *
import random
import multiprocessing as mp
import os

def interact(des,n):
    proc=call('C://Users//Tina//Desktop//main.exe DRLNEW '+des+' '+n,shell=True)
    print(proc)


def job():
    for i in range(10):
        des="No."+str(random.randint(1,10))
        n=str(random.randint(1,100))
        interact(des,n)


if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p = mp.Pool(4)
    for i in range(5):
        p.apply_async(job,args=(i,))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')

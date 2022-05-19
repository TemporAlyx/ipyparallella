import ipyparallel as ipp
from functools import partial
import numpy as np
import gc
import logging

import psutil
n_cpus = psutil.cpu_count() // 2

is_initialized = False
cluster = None
lview = None
dview = None

def initialize(n_cpus=n_cpus, objs=[]):
    global is_initialized, cluster, lview, dview
    cluster = ipp.Cluster(n=n_cpus, log_level=logging.ERROR)
    cluster.start_cluster_sync()

    rc = cluster.connect_client_sync()
    rc.wait_for_engines(n_cpus, interactive=False)
    # print(rc.ids)

    lview = rc.load_balanced_view()
    lview.block = True

    dview = rc[:]
    dview.block = True

    is_initialized = True
    init_imports()

    if objs:
        push(objs)


def init_imports():
    global is_initialized, dview

    if is_initialized:
        dview.execute('import numpy')
        dview.execute('np = numpy')
        dview.execute('import pandas')
        dview.execute('pd = pandas')
        dview.execute('from scipy import stats')
        dview.execute('import math')
        dview.execute('import time')
        dview.execute('import random')
    else:
        print('Error: cannot import, cluster not initialized')

def shutdown():
    global is_initialized, cluster, lview, dview
    if is_initialized:
        cluster.stop_cluster_sync()
        is_initialized = False
    cluster = None
    lview = None
    dview = None

def restart():
    shutdown()
    initalize()

def push(objs):
    global is_initialized, dview

    if is_initialized:
        if type(objs) is list:
            for obj in objs:
                if type(obj) is dict:
                    dview.push(obj)
                elif type(obj) is str:
                    dview.execute(obj)
        elif type(objs) is dict:
            dview.push(objs)
        elif type(objs) is str:
            dview.execute(objs)
        else:
            print('Error: format must be dict of objs, str for execute, or list of either')
    else:
        print('Error: cannot push objects, cluster not initialized')

def apply(func, inputs, objs=[], n_cpus=n_cpus, d=False, stagger=None): # *params, 
    global is_initialized, lview, dview
    was_local_init = False

    if not is_initialized:
        was_local_init = True
        initialize(n_cpus, objs)

    # if params:  # an attempt at including additional args
    #     afunc = partial(func, params)  #input function could instead be lambdaed 
    # else:
    #     afunc = func
    if stagger is None:
        if d:
            outputs = dview.map(func, inputs)
        else:
            outputs = lview.map(func, inputs)
    else:
        inputs = np.array(inputs)
        outputs = np.empty(len(inputs), dtype=object)
        x = 0
        while x+stagger < len(inputs):
            if d:
                outputs[x:x+stagger] = dview.map(func, inputs[x:x+stagger])
            else:
                outputs[x:x+stagger] = lview.map(func, inputs[x:x+stagger])
        if d:
            outputs[x:] = dview.imap(func, inputs[x:])
        else:
            outputs[x:] = lview.imap(func, inputs[x:])


    if was_local_init:
        shutdown()

    return outputs
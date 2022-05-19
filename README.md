# ipyparallella
A hacky wrapper for ipyparallel that allows for one-liner multi-processing
created largely out of sheer laziness, and also frusteration at how borked multiprocessing is in ipython on windows

Requires ipyparallel to be installed
additionally, by default it initializes clusters with numpy, pandas, and scipy.stats, and so these need to be installed as well

# Basic Usage

place ipyparallella.py in project folder

```python
import time
import ipyparallella as ipla

def parallel_function(inputs):
    time.sleep(1)  # do something crazy
    return inputs
    
outputs = ipla.apply(parallel_function, range(120))
print(outputs)  # one-liner without initialization, incurs ~0.5-1s of overhead per core in cluster
```

without intitalization it defaults to using half your cpu cores (psutil.cpu_count() // 2), initializes a cluster, performs the operations, and then shutsdown the cluster.

```python
ipla.initialize(n_cpus=12)  # alternatively you can pre-initialze and then use the cluster for multiple function passes

outputs = ipla.apply(parallel_function, range(120)
print(outputs)

ipla.shutdown() 
```

leaving a cluster running in the background doesn't *generally* cause any problems, but for extended use over many functions and several hours, sometimes things run smoother if the cluster is restarted every now and again, hence in sparse use cases, letting the one-liner self-initialize and shutdown is simpler and safer.

```python
def sleep():
    time.sleep(1)

def parallel_function(inputs):
    sleep()  # if the paralleized function requires external functions or libraries outside of the default initialization
    return inputs  # then these need to be provided as a dict of these objects
    
outputs = ipla.apply(parallel_function, range(120), objs={'sleep': sleep})
print(outputs)
```

# To Do

* add notebook detailing use cases and features
* build in catches for common errors for easier bug-fixing
* change default imports to read from a file for easier editing
* allow passing a list containing multiple dicts of objects to push, as well as strings of packages to import
* assess difficulty in listing this as a proper python package

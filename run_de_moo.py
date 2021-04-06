# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 14:15:54 2021

@author: robin
"""
import numpy as np
from MOORobin.MOO import MOO
from smt.problems import Rosenbrock, Branin
from smt.sampling_methods import LHS
from smt.surrogate_models import LS, QP, KPLS, KRG, KPLSK, GEKPLS, MGP
from smt.utils import compute_rms_error

#%%problem definition
ndim = 2
ny = 2
fun1 = Rosenbrock(ndim=ndim)
fun2 = Branin(ndim=ndim)

def objective(x):
    return [fun1(x), fun2(x)]
    
xlimits = np.array([[-2.0,2.0], [-2.0,2.0]])

mo = MOO(n_iter = 2, xlimits = xlimits, n_gen=30, pop_size = 30)
#%%
from pymoo.visualization.scatter import Scatter
mo.optimize(objective)
res = mo.result
plot = Scatter()
plot.add(res.F, color="red")
plot.show()


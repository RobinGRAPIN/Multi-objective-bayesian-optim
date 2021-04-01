# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 14:08:54 2021

@author: robin
"""

import sys
sys.path.insert(0,'C:/Users/robin/bayesian-optim')

#%% imports

import numpy as np
from random import randint
import matplotlib.pyplot as plt

from smt.problems import Rosenbrock, Branin
from smt.sampling_methods import LHS
from smt.surrogate_models import LS, QP, KPLS, KRG, KPLSK, GEKPLS, MGP
from smt.utils import compute_rms_error


from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.problem import Problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

#%% Optimization loop incrementing the surrogates

"""
Pour des problèmes où f serait très chère à évaluer on va
réduire ndoe (nombre de points du maillage) et évaluer de nouvaux points
au fur et à mesure de l'optimisation des modèles
"""
ndim = 2
ny = 2
fun1 = Rosenbrock(ndim=ndim)
fun2 = Branin(ndim=ndim)

def objective(x):
    return [fun1(x), fun2(x)]
    
xlimits = np.array([[-2.0,2.0], [-2.0,2.0]])

def minimise(objective, xlimits, algo=NSGA2(pop_size=100) , ngen=50, 
             verbose=False, seed=1, itermax = 10, precision = 1e-5,
              ndoe = 10): #sampling = 'LHS'
    n_iter = 0    
    
    # Construction of the DOE    
    sampling = LHS(xlimits=xlimits)
    xt = sampling(ndoe)
    yt = [fun1(xt), fun2(xt)]
    
    #training the model
    liste=[]
    for iny in range(ny):
        t= KRG(theta0=[1e-2]*ndim,print_prediction = False)
        t.set_training_values(xt,yt[iny])    
        t.train()
        liste.append(t)
    
    #creation of the pymoo problem on the surrogate model
    class prob(Problem):

        def __init__(self):
            super().__init__(n_var=2,
                             n_obj=2,
                             n_constr=0,
                             xl=np.array([-2.0, -2.0]),
                             xu=np.array([2.0, 2.0]),
                             elementwise_evaluation=True)
    
        def _evaluate(self, x, out, *args, **kwargs):
            xx = np.asarray(x).reshape(1, -1) 
            f1 = liste[0].predict_values(xx)[0][0]
            f2 = liste[1].predict_values(xx)[0][0]
            out["F"] = [f1, f2]
    
    problem = prob()

    #model incrementation loop
    while (n_iter < itermax):
        n_iter += 1
        print("iteration ",n_iter)
        
        #minimize current model with GA
        res = minimize(problem,algo,("n_gen",ngen),verbose=False,seed=seed)
        
        #choice of a point of the set for convergence or to be the next one
        new_x, y_model = select_point(res)
              
        #eval of f in this point
        new_y = objective(np.array([new_x]))
        
        #stop criteria on precision (or smthng else)
        #if np.linalg.norm(new_y - y_model) < precision :
         #   return res
        
        #update model with the new point
        for i in range(len(yt)):            
            yt[i] = np.atleast_2d(np.append(yt[i],new_y[i],axis=0))
        xt = np.atleast_2d(np.append(xt,np.array([new_x]),axis=0))

        for iny in range(ny):
            t= KRG(theta0=[1e-2]*ndim,print_prediction = False,print_global=False)
            t.set_training_values(xt,yt[iny])        
            t.train()
            liste[iny]=t
        
    #run GA to the best model
    return minimize(problem,algo,("n_gen",ngen),verbose=False,seed=seed)
        
        
def select_point(result): #pour l'instant j'en prends juste un au hasard
    # à terme faire en fonction de leur distance ? ou autre ?
    X = result.X
    Y = result.F
    i = randint(0,X.shape[0])
    return X[i,:], Y[i,:]

#%% test and visualization

resultat= minimise(objective, xlimits,ngen=20, algo=NSGA2(pop_size=50))

#visualisation of the Pareto front
plot = Scatter()
plot.add(resultat.F, color="red")
plot.show()


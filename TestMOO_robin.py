# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 10:30:51 2021

@author: robin
"""

import sys
sys.path.insert(0,'C:/Users/robin/bayesian-optim')

#%% imports

import numpy as np
from smt.problems import Rosenbrock, Branin
from smt.sampling_methods import LHS
from smt.surrogate_models import LS, QP, KPLS, KRG, KPLSK, GEKPLS, MGP
import matplotlib.pyplot as plt
from smt.utils import compute_rms_error
from random import randint

#%% MOO problem definition

ndim = 2
ndoe = 20 
ny = 2
fun1 = Rosenbrock(ndim=ndim)
fun2 = Branin(ndim=ndim)

# Construction of the DOE
xlimits = np.array([[-2.0,2.0], [-2.0,2.0]])
sampling = LHS(xlimits=xlimits)
xt = sampling(ndoe)
yt = [fun1(xt), fun2(xt)]

# Construction of the validation points
ntest = 100 
sampling = LHS(xlimits=xlimits)
xtest = sampling(ntest)
ytest = [fun1(xtest), fun2(xtest)]

#%% Modelization

list_t=[]
########### The Kriging model
for iny in range(ny):
    print('Output ', iny)
    # The variable 'theta0' is a list of length ndim.
    t= KRG(theta0=[1e-2]*ndim,print_prediction = False)
    t.set_training_values(xt,yt[iny])

    t.train()
    list_t.append(t)

    # Prediction of the validation points
    y = t.predict_values(xtest)
    print('Kriging,  err: '+ str(compute_rms_error(t,xtest,ytest[iny])))
    
print("theta values for output ", 0, " = ",  list_t[0].optimal_theta)
print("theta values for output ", 1, " = ",  list_t[1].optimal_theta)

#%% Pymoo imports

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.problem import Problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

#%% Optimization

class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         n_constr=0,
                         xl=np.array([-2.0, -2.0]),
                         xu=np.array([2.0, 2.0]),
                         elementwise_evaluation=True)

    def _evaluate(self, x, out, *args, **kwargs):
        xx = np.asarray(x).reshape(1, -1) #le modèle prend un array en entrée
        f1 = list_t[0].predict_values(xx)[0][0]
        f2 = list_t[1].predict_values(xx)[0][0]
        out["F"] = [f1, f2]
        #out["G"] = [g1, g2]

problem = MyProblem()

algorithm = NSGA2(pop_size=100)

res = minimize(problem,
               algorithm,
               ("n_gen", 100),
               verbose=True,
               seed=1)

plot = Scatter()
plot.add(res.F, color="red")
plot.show()

#%% real pareto front

class MyProblem_reel(Problem):

    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         n_constr=0,
                         xl=np.array([-2.0, -2.0]),
                         xu=np.array([2.0, 2.0]),
                         elementwise_evaluation=True)

    def _evaluate(self, x, out, *args, **kwargs):
        xx = np.asarray(x).reshape(1, -1) #le modèle prend un array en entrée
        f1 = fun1(xx)[0][0]
        f2 = fun2(xx)[0][0]
        out["F"] = [f1, f2]
        #out["G"] = [g1, g2]

problem_exact = MyProblem_reel()

algorithm_bis = NSGA2(pop_size=100)

res_exact = minimize(problem_exact,
               algorithm_bis,
               ("n_gen", 100),
               verbose=True,
               seed=1)

plot = Scatter()
plot.add(res_exact.F, color="red")
plot.show()
# résultat très similaire au plot précédent

#%% test zone
def funfun(x, out, *args, **kwargs):
    out["F"] = [1,1]
    
problem._evaluate = funfun

res = minimize(problem,
               algorithm,
               ("n_gen", 100),
               verbose=True,
               seed=1)


#%% Optimization loop incrementing the surrogates

"""
Pour des problèmes où f serait très chère à évaluer on va
réduire ndoe (nombre de points du maillage) et évaluer de nouvaux points
au fur et à mesure de l'optimisation des modèles

On part du modèle à 20 points de sampling précédent, qu'on raffinera à
chaque exécution de l'algorithme génétique'
"""


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
        f1 = list_t[0].predict_values(xx)[0][0]
        f2 = list_t[1].predict_values(xx)[0][0]
        out["F"] = [f1, f2]   


def minimise(problem, algo, ngen=100, verbose=False,
             seed=1, itermax = 10, precision = 1e-5):
    n_iter = 0
    
    while (n_iter < itermax):
        n_iter += 1
        print("iteration "+n_iter)
        
        #minimize current model
        res = minimize(problem,algo,("n_gen",ngen),verbose=False,seed=seed)
        
        #choice of a point of the set for convergence or to be the next one
        new_x, y_model = select_point(res)
              
        #eval of f in this point
        new_y = [fun1(new_x),fun2(new_x)]
        
        #stop criteria on precision (or smthng else)
        if np.linalg.norm(np.asarray(new_y) - y_model) < precision :
            break
        
        #if not stop criteria update model with new point
        
        #and update problem._evaluate
        def neweval(x, out, *args, **kwargs):
            xx = np.asarray(x).reshape(1, -1) 
            f1 = new_model_f1.predict_values(xx)[0][0]
            f2 = new_model_f2.predict_values(xx)[0][0]
            out["F"] = [f1, f2]   
        problem._evaluate = neweval
        
        
    #do we run one last time the GA as it is cheap and we have a new point ?
    return 
        
        
def select_point(result): #pour l'instant j'en prends juste un au hasard
    # à terme faire en fonction de leur distance ou autre ?
    X = res.X
    Y = res.F
    i = randint(X.shape[0])
    return X[i,:], Y[i,:]

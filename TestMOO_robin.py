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

#%% Pymoo imports

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.problem import Problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

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

#%% Real Pareto front

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
# résultat très similaire au plot précédent avec 100 gen de 100

#%% multi-dimensional activation function

from scipy.stats import norm

#pour l'instant qu'en dim 2...en fait sert a rien je dois faire direct sa dérivée ?
def PI(x, pareto_front , moyennes, sigma ):
    """
    Parameters
    ----------
    x : list
        coordonnées du point à évaluer.
    pareto_front : liste
        liste des valeurs dans l'espace objectif des points optimaux 
        du modèle actuel.
    moyennes : list
        liste des fonctions moyennes du modèle par objectif à approximer.
    sigma : list
        liste des variances des modèles sur chaque objectif.

    Returns
    -------
    pi_x : float
        PI(x) : probabilité que x soit une amélioration.
    """
    m = len(pareto_front)
    pi_x = norm.cdf((pareto_front[0][0] - moyennes[0](x))/sigma[0](x))
    for i in range(1,m):
        pi_x += ((norm.cdf((pareto_front[0][i+1] - moyennes[0](x))/sigma[0](x))
                 - norm.cdf((pareto_front[0][i] - moyennes[0](x))/sigma[0](x)))
                 * norm.cdf((pareto_front[1][i+1] - moyennes[1](x))/sigma[1](x)))
    pi_x += (1 - norm.cdf((pareto_front[0][m] - moyennes[0](x))/sigma[0](x)))*norm.cdf((pareto_front[1][m] - moyennes[1](x))/sigma[1](x))
    return pi_x

def best_points(Y):
    index = [] #indices of the best points (Pareto)
    n = len(Y)
    dominated = [False]*n
    for y in range(n):
        if not dominated[y]:
            for y2 in range(y+1,n):
                if not dominated[y2]:#if y2 is dominated (by y0), we already compared y0 to y
                    y_domine_y2 , y2_domine_y = dominate_min(Y[y],Y[y2])
                    
                    if y_domine_y2 :
                        dominated[y2]=True
                    if y2_domine_y :
                        dominated[y]=True
                        break
            if not dominated[y]:
                index.append(y)
    return index
                    
# retourne a-domine-b , b-domine-a !! for minimization !!  
def dominate_min(a,b):
    a_bat_b = False
    b_bat_a = False
    for i in range(len(a)):
        if a[i] < b[i]:
            a_bat_b = True
            if b_bat_a :
                return False, False # = equivalent points
        if a[i] > b[i]:
            b_bat_a = True
            if a_bat_b :
                return False, False    
    if a_bat_b and (not b_bat_a):
        return True, False
    if b_bat_a and (not a_bat_b):
        return False, True
    return False, False # égaux

#%% testons
Y = [ [6,9,3] , [1,1,11],[4,4,2]]
print(dominate_min(Y[0], Y[1]))#false, FAlse
print(dominate_min(Y[0], Y[2]))#false, true
print( best_points(Y))# 1,2

#with arrays
Y = [np.asarray(i) for i in  Y]
print(dominate_min(Y[0], Y[1]))#false, FAlse
print(dominate_min(Y[0], Y[2]))#false, true
print( best_points(Y))# 1,2

#%% test PI







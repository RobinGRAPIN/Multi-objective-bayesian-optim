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

#%% 1st MOO problem definition

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

#%% Optimization through model

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

#%% 1st problem exact Pareto front

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

#pour l'instant qu'en dim 2
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
        PI(x) : probabilité que x soit une amélioration € [0,1]
    """
    m = len(pareto_front)
    try :
        pi_x = norm.cdf((pareto_front[0][0] - moyennes[0](x))/sigma[0](x))
        for i in range(1,m):
            pi_x += ((norm.cdf((pareto_front[0][i+1] - moyennes[0](x))/sigma[0](x))
                     - norm.cdf((pareto_front[0][i] - moyennes[0](x))/sigma[0](x)))
                     * norm.cdf((pareto_front[1][i+1] - moyennes[1](x))/sigma[1](x)))
        pi_x += (1 - norm.cdf((pareto_front[0][m] - moyennes[0](x))/sigma[0](x)))*norm.cdf((pareto_front[1][m] - moyennes[1](x))/sigma[1](x))
        return pi_x
    except : #for training points having sigma = 0
        return 0

def best_points(Y):
    index = [] #indexes of the best points (Pareto)
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
                return False, False # same front
        if a[i] > b[i]:
            b_bat_a = True
            if a_bat_b :
                return False, False    
    if a_bat_b and (not b_bat_a):
        return True, False
    if b_bat_a and (not a_bat_b):
        return False, True
    return False, False # same points

# testons
Y = [ [6,9,3] , [1,1,11],[4,4,2]]
print(dominate_min(Y[0], Y[1]))#false, False
print(dominate_min(Y[0], Y[2]))#false, true
print( best_points(Y))# 1,2

#with arrays
Y = [np.asarray(i) for i in  Y]
print(dominate_min(Y[0], Y[1]))#false, False
print(dominate_min(Y[0], Y[2]))#false, true
print( best_points(Y))# 1,2

aa = np.array([[2.,2.]])
#print([i.predict_values(aa)[0][0] for i in list_t])

Yy = [[0.45    ,   5.88251506],
      [0.55    ,   0.55697145],#
      [0.65    ,   2.50759323],
      [0.75    ,   6.87371526],
      [0.25    ,   5.54137477],
      [0.35    ,   3.72052642],
      [0.05    ,   2.84688711],#
      [0.85    ,   0.9366706 ],
      [0.15    ,   5.00527782],
      [0.95    ,   5.78338178]]
plt.scatter(np.transpose(np.asarray(Yy))[0],np.transpose(np.asarray(Yy))[1])
plt.show()
print(best_points(Yy))#seems good 1,6

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


#%% use of MOO module for 1st example
from MOORobin.MOO import MOO
ndim = 2
ny = 2
fun1 = Rosenbrock(ndim=ndim)
fun2 = Branin(ndim=ndim)
xlimits = np.array([[-2.0,2.0], [-2.0,2.0]])
def objective(x):
    return [fun1(x), fun2(x)]

#setup
class MyProblem_reel(Problem):
    def __init__(self):
        super().__init__(n_var=2,n_obj=2,n_constr=0,xl=np.array([-2.0, -2.0]),xu=np.array([2.0, 2.0]),elementwise_evaluation=True)
    def _evaluate(self, x, out, *args, **kwargs):
        xx = np.asarray(x).reshape(1, -1) #le modèle prend un array en entrée
        f1 = fun1(xx)[0][0]
        f2 = fun2(xx)[0][0]
        out["F"] = [f1, f2]
problem_exact = MyProblem_reel()
moga = MOO(n_iter = 15,criterion = "GA", n_start = 15, xlimits = xlimits, n_gen=30, pop_size = 30)
mopi = MOO(n_iter = 15,criterion = "PI", n_start = 15, xlimits = xlimits, n_gen=30, pop_size = 30)

# run & plots
plots_ = []
res_exact = minimize(problem_exact,NSGA2(pop_size=100),("n_gen", 100),verbose=False)
moga.optimize(objective)
mopi.optimize(objective)
plots_.append( np.array(moga.result.F[:]) )
plots_.append( np.array(mopi.result.F[:]) )
plots_.append( np.array(res_exact.F[:]) )
plt.scatter(plots_[0][:,0], plots_[0][:,1], c = "green", label = "GA")
plt.xlabel("f1 = Rosenbrock")
plt.ylabel("f2 = Branin")
plt.scatter(plots_[1][:,0], plots_[1][:,1], c = "red", label = "PI")
plt.scatter(plots_[2][:,0], plots_[2][:,1], c = "blue", label = "True")
plt.title("Methods comparation, niter = nstart = 15")
plt.legend()
plt.show()

#%% ZDT with GA
from MOORobin.MOO import MOO
    
xlimits = np.array([[0.,1.], [0.,1.]])

mo = MOO(n_iter = 20,n_start = 20, xlimits = xlimits, criterion = "GA",n_gen=30, pop_size = 100)

# ZDT
import numpy as np
from MOORobin.zdt import ZDT
funf1 = ZDT()
funf2 = ZDT(type=2)
funf3 = ZDT(type=3)
#funf4 = ZDT(type=4)
#funf5 = ZDT(type=5)
tests = [funf1,funf2,funf3]#,funf4,funf5]
#testpt = np.array([[1,1],[1,1.2]])
#print(funfun(testpt))
from pymoo.visualization.scatter import Scatter
plots = []
for i in range(len(tests)):
    mo.optimize(tests[i])
    res = mo.result
    plots.append( np.array(mo.result.F[:]) )
plt.scatter(plots[0][:,0], plots[0][:,1], c = "green", label = "ZDT1")
plt.xlabel("f1")
plt.ylabel("f2")
plt.scatter(plots[1][:,0], plots[1][:,1], c = "red", label = "ZDT2")
plt.scatter(plots[2][:,0], plots[2][:,1], c = "blue", label = "ZDT3")
plt.legend()
plt.title("GA MOBOpt method, n_iter = 20, n_init = 20, ngen = 30, pop_size = 100")
plt.xlim([0,1])
plt.ylim([-1,1])
plt.show()

#%%ZDT with PI

from MOORobin.MOO import MOO
    
xlimits = np.array([[0.,1.], [0.,1.]])

mo_pi = MOO(n_iter = 2,n_start = 10, xlimits = xlimits, criterion = "PI", n_gen = 30,pop_size=30)

plots_pi = []
for i in range(len(tests)):
    mo_pi.optimize(tests[i])
    res = mo_pi.result
    titre = "ZDT "+str(i+1)+ " using PI"
    plots_pi.append( np.array(mo.result.F[:]) )
plt.scatter(plots_pi[0][:,0], plots_pi[0][:,1], c = "green", label = "ZDT1")
plt.xlabel("f1")
plt.ylabel("f2")
plt.scatter(plots_pi[1][:,0], plots_pi[1][:,1], c = "red", label = "ZDT2")
plt.scatter(plots_pi[2][:,0], plots_pi[2][:,1], c = "blue", label = "ZDT3")
plt.legend()
plt.title("Methode PI, n_iter = n_init = 10")
#plt.xlim([0,1])
#plt.ylim([-1,1])
plt.show()

#%% ZDT exact Pareto front

from MOORobin.zdt import ZDT
funf1 = ZDT()
funf2 = ZDT(type=2)
funf3 = ZDT(type=3)
tests = [funf1,funf2,funf3]
plots = []
for i in range(len(tests)):
    print("optimizing zdt ", i+1)
    objectives = tests[i]
    class MyProblem_reel(Problem):

        def __init__(self):
            super().__init__(n_var=2,
                             n_obj=2,
                             n_constr=0,
                             xl=np.array([0.,0.]),
                             xu=np.array([1.0, 1.0]),
                             elementwise_evaluation=True)
    
        def _evaluate(self, x, out, *args, **kwargs):
            xx = np.asarray(x).reshape(1, -1) #le modèle prend un array en entrée
            out["F"] = [objectives(xx)[0][0][0], objectives(xx)[1][0][0]]
    zdt_exact = MyProblem_reel()
    algorithm_bis = NSGA2(pop_size=100)    
    res_exact = minimize(zdt_exact,algorithm_bis,("n_gen", 100))
    plots.append( res_exact.F )

plt.scatter(plots[0][:,0], plots[0][:,1], c = "green", label = "ZDT1")
plt.xlabel("f1")
plt.ylabel("f2")
plt.scatter(plots[1][:,0], plots[1][:,1], c = "red", label = "ZDT2")
plt.scatter(plots[2][:,0], plots[2][:,1], c = "blue", label = "ZDT3")
plt.legend()
plt.title("Expected Pareto front")
#plt.xlim([0,1])
#plt.ylim([0,1])
plt.show()


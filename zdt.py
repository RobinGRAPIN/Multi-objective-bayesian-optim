# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 10:07:51 2021

@author: robin
"""

#%% fonction test relou

"""
ZDT toolkit
x = {x1.. xn}
y = {x1.. xj} = {y1.. yj} #for simplicity, here, j = Int(n/2)
z = {x(j+1).. xn} = {z1.. zk}
fonctions test de la forme :
    f1 : y -> f1(y)
    f2 : y,z -> g(z)h(f1(y),g(z))
xbounds = [0,1]**n
"""

import numpy as np

from smt.problems.problem import Problem


class ZDT(Problem):
    def _initialize(self):
        self.options.declare("ndim", 2, types=int)
        self.options.declare("name", "ZDT", types=str)
        self.options.declare("type",1,values=[1,2,3,4,5], type =int)#one of the 5 test functions
        
    def evaluate(self, x):
        """
        Arguments
        ---------
        x : ndarray[ne, n_dim]
            Evaluation points.

        Returns
        -------
        [ndarray[ne, 1],ndarray[ne, 1]]
            Functions values.
        """
        ne, nx = x.shape
        #cutting x into y and z
        j = int(nx/2)
        
        f1 = np.zeros((ne,1))
        f2 = np.zeros((ne,1))
        
        if self.options["type"] <5 :
            f1 = x[:,0]
        else :
            f1 = 1 - np.exp(-4*x[:,0]) * np.sin(6*np.pi*x[:,0])**6
        
        #g
        g = np.zeros((ne,1))
        if self.options["type"] < 4:
            for i in range(ne):
                g[i,0] = 1 + 9 / (nx -j) * sum(x[i, j+1 : nx])
        elif self.options["type"] == 4:
            for i in range(ne):
                g[i,0] = 1 + 10*(nx -j)+ sum(x[i, j+1 : nx]**2 - 10*np.cos(4*np.pi*x[i, j+1 : nx]))
        else :
            for i in range(ne):
                g[i,0] = 1 + 9 / (nx -j) * sum(x[i, j+1 : nx])**0.25
                
        #h, more exactly : f2 = g * h
        if self.options["type"] ==1 or self.options["type"] ==4:
            for i in range(ne):
                f2[i,0] = g[i,0]*( 1 -np.sqrt(f1[i,0]/g[i,0]))
        elif self.options["type"] == 2 or self.options["type"] ==5:
            for i in range(ne):
                f2[i,0] = g[i,0]*( 1 - (f1[i,0]/g[i,0])**2 )                
        else :
            for i in range(ne):
                f2[i,0] = g[i,0]*( 1 -np.sqrt(f1[i,0]/g[i,0]) - f1[i,0]/g[i,0]*np.sin(10*np.pi*f1[i,0]))
       
        return [f1,f2]
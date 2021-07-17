################# COMPREHENSIVE_LEARNING_PARTICLE_SWARM_OPTIMIZATION-(CLPSO) #############
# ***************************************************************************************#
# Author:  Ph.D Thu Huynh Van, Assist. Prof. Sawekchai Tangaramvong 
# Emails:  thuxd11@gmail.com, Sawekchai.T@chula.ac.th
#          Applied Mechanics and Structures Research Unit, Department of Civil Engineering, 
#          Chulalongkorn University
""" Research paper: Two-Phase ESO-CLPSO Method for the Optimal Design of Structures 
with Discrete Steel Sections (2021) """
# ***************************************************************************************#
import random
import math
import numpy as np
from numpy.linalg import norm
from scipy.linalg import eigh
import matplotlib.pyplot as plt 
import sys
import scipy
import itertools

#--- COST FUNCTION 
# function we are attempting to optimize (minimize)
def func1(x):
     return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
 
#--- MAIN 
class Particle:
    def __init__(self,x0):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.pos_best_i_record=[]   # particles x dimensions NP*D
        self.err_best_i= 1000000    # best error individual
        self.err_best_i_record=[]
        self.err_i=-1               # error individual
        for i in range(0,num_dimensions):
            self.velocity_i.append(random.uniform(-1,1)*(bounds[i][1]-bounds[i][0])+bounds[i][0])
            self.position_i.append(random.uniform(-1,1)*(bounds[i][1]-bounds[i][0])+bounds[i][0])
        self.pos_best_i_record.append(self.position_i)
        
    # evaluate current fitness
    def evaluate(self, costFunc):
        self.err_i = costFunc(self.position_i)  
        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i: 
            self.pos_best_i = self.position_i   
            self.err_best_i = self.err_i     
            
    # update new particle velocity
    def update_velocity(self, pbest_f, pos_best_g, bounds, mdblI):
        c1=2        # cognative constant
        c2=2        # social constant
        for i in range(0, num_dimensions):
            r1=random.random()
            r2=random.random()
            vel_cognitive = c1*r1*(pbest_f[i] - self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i] = mdblI*self.velocity_i[i] + vel_cognitive + vel_social
            self.velocity_i[i] = np.where(self.velocity_i[i] >= 0.2*(bounds[i][1]-bounds[i][0]), 0.2*(bounds[i][1]-bounds[i][0]), self.velocity_i[i]).tolist()
    
    # update the particle position based off new velocity updates
    def update_position(self, bounds):
        for i in range(0, num_dimensions):
            self.position_i[i] = self.position_i[i] + self.velocity_i[i]
            # adjust maximum position if necessary
            if self.position_i[i] > bounds[i][1]:
                self.position_i[i] = bounds[i][1]
            # adjust minimum position if neseccary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i] = bounds[i][0]
                
class CLPSO():
    def __init__(self,costFunc,x0,bounds,num_particles,maxiter):
        
        global num_dimensions
        num_dimensions = len(x0); self.err_best_g = 10000000; self.pos_best_g = []                   
        self.pos_best_g_record = []; self.pos_best_g_record_1 = []
        self.err_best_g_record = []
        self.global_best_all_iteration = []; self.iteration = []
        self.f_pbest = []; self.pbest_f = []; self.pbest_f_1 = []; self.Pc = []
        fi1 = [0]*num_dimensions; fi2 = [0]*num_dimensions; fi = [0]*num_dimensions
        bi1 = 0; self.bi = [0]*num_dimensions; self.mintSinceLastChange = [0]*num_particles
        mintNuC = 5; mdblI = 0; self.best_position = 0
        
        # Calculate learning probability Pc
        t=np.linspace(0,5,num_particles)
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))   
            self.Pc.append(0+0.5*(np.exp(t[i])-np.exp(t[0]))/(np.exp(5)-np.exp(t[0])))
            self.f_pbest.append([i]*num_dimensions)
            
        for k in range(0,num_particles):
            swarm[k].evaluate(costFunc)
            if swarm[k].err_i < self.err_best_g:  
                self.pos_best_g = list(swarm[k].position_i)
                self.err_best_g = float(swarm[k].err_i)
            self.pos_best_g_record.append(swarm[k].pos_best_i)
            self.err_best_g_record.append(swarm[k].err_best_i)
            self.pbest_f.append(swarm[k].position_i)
            
        for v in range(0, num_particles):
            for z in range(0,num_dimensions): 
                 fi1[z] = math.ceil(random.uniform(0,1)*(num_particles-1))
                 fi2[z] = math.ceil(random.uniform(0,1)*(num_particles-1))
                 fi[z] = np.where(self.err_best_g_record[fi1[z]] < self.err_best_g_record[fi2[z]], fi1[z], fi2[z]).tolist()
                 bi1 = random.random() - 1 + self.Pc[v]
                 self.bi[z] = np.where(bi1 >= 0, 1, 0).tolist()
            if np.sum(self.bi) == 0:
                rc = round(random.uniform(0,1)*(num_dimensions-1))
                self.bi[rc] = 1
            for m in range(0,num_dimensions):
                self.f_pbest[v][m] = self.bi[m]*fi[m] + (1-self.bi[m])*self.f_pbest[v][m]
                
        i=0
        while i < maxiter:
            self.iteration.append(i)
            self.pos_best_g_record_1 = np.copy(self.pos_best_g_record)
            self.pbest_f_1=np.copy(self.pbest_f)
            
            for j in range(0,num_particles):
                 if self.mintSinceLastChange[j] > mintNuC:
                    self.mintSinceLastChange[j] = 0
                    
                    for k in range(0,num_dimensions):
                         fi1[k] = math.ceil(random.uniform(0,1)*(num_particles-1))
                         fi2[k] = math.ceil(random.uniform(0,1)*(num_particles-1))
                         fi[k] = np.where(self.err_best_g_record[fi1[k]] < self.err_best_g_record[fi2[k]], fi1[k], fi2[k]).tolist()
                         bi1 = random.uniform(0,1) - 1 + self.Pc[j]
                         self.bi[k] = np.where(bi1>=0, 1, 0).tolist()
                    if np.sum(self.bi) == 0:
                         rc = round(random.uniform(0,1)*(num_dimensions-1))
                         self.bi[rc] = 1
                    for m in range(0,num_dimensions):
                         self.f_pbest[j][m] = self.bi[m]*fi[m] + (1-self.bi[m])*self.f_pbest[j][m] 
                         
            for j in range(0,num_particles):
                 for k in range(0,num_dimensions):
                         index_1 = self.f_pbest[j][k]
                         self.pbest_f_1[j,k] = self.pos_best_g_record_1[index_1, k]
            self.pbest_f = self.pbest_f_1.tolist()
            
            # cycle through swarm and update velocities and position
            mdblI = 0.9 - (0.9 - 0.4) * i / maxiter
            for j in range(0,num_particles):
                swarm[j].update_velocity(self.pbest_f[j], self.pos_best_g, bounds, mdblI)
                swarm[j].update_position(bounds)
                
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)
                # Update the personal best position and fitness values for population
                if swarm[j].err_i < self.err_best_g_record[j]:
                    self.pos_best_g_record[j] = list(swarm[j].position_i)
                    self.err_best_g_record[j] = float(swarm[j].err_i)
                else:
                    self.mintSinceLastChange[j] += 1
                # determine if current particle is the best (globally)
                if swarm[j].err_i < self.err_best_g: 
                    self.pos_best_g = list(swarm[j].position_i)
                    self.err_best_g = float(swarm[j].err_i)
            self.global_best_all_iteration.append(self.err_best_g)
            i+=1
            
        self.best_position = self.pos_best_g
        
        # Final results
        print('FINAL RESULTS:', self.pos_best_g, '---', self.err_best_g)
        
initial=[-1, 1]   # initial starting location [x1,x2...]
bounds=[(-2.08, 2.08),(-2.08, 2.08)] 
t = CLPSO(func1, initial, bounds, num_particles=10, maxiter=500)

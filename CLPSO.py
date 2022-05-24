################# COMPREHENSIVE_LEARNING_PARTICLE_SWARM_OPTIMIZATION-(CLPSO) #############
# ***************************************************************************************#
""" Author:  Ph.D Thu Huynh Van, Assoc. Prof. Sawekchai Tangaramvong 
#   Emails:  thuxd11@gmail.com, Sawekchai.T@chula.ac.th
#            Applied Mechanics and Structures Research Unit, Department of Civil Engineering, 
#            Chulalongkorn University """
# Research paper: Two-Phase ESO-CLPSO Method for the Optimal Design of Structures 
# with Discrete Steel Sections (2021) """
# ***************************************************************************************#
import random
import math
import numpy as np

#--- COST FUNCTION 
# Function we are attempting to optimize (minimize)
def func1(x):
     return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
 
#--- MAIN 
class Particle:
    def __init__(self):
        self.position_i = []          # particle position
        self.velocity_i = []          # particle velocity
        self.pos_best_i = []          # best position individual
        self.pos_best_i_record = []   # particles x dimensions NP*D
        self.err_best_i = 1000000     # best error individual
        self.err_best_i_record = []
        self.err_i = -1               # error individual
        for i in range(0, num_dimensions):
            self.velocity_i.append(random.uniform(-1,1)*(bounds[i][1] - bounds[i][0]) + bounds[i][0])
            self.position_i.append(random.uniform(-1,1)*(bounds[i][1] - bounds[i][0]) + bounds[i][0])
        self.pos_best_i_record.append(self.position_i)
        
    # Evaluate current fitness
    def evaluate(self, costFunc):
        self.err_i = costFunc(self.position_i)
        # Check to see if the current position is an individual best
        if self.err_i < self.err_best_i: 
            self.pos_best_i = self.position_i   
            self.err_best_i = self.err_i     
            
    # Update new particle velocity
    def update_velocity(self, pbest_f, pos_best_g, bounds, mdblI):
        c1 = 2        # cognative constant
        c2 = 2        # social constant
        for i in range(0, num_dimensions):
            r1 = random.random()
            r2 = random.random()
            vel_cognitive = c1*r1*(pbest_f[i] - self.position_i[i])
            vel_social = c2*r2*(pos_best_g[i] - self.position_i[i])
            self.velocity_i[i] = mdblI*self.velocity_i[i] + vel_cognitive + vel_social
            self.velocity_i[i] = np.where(self.velocity_i[i] >= 0.2*(bounds[i][1] - bounds[i][0]), 0.2*(bounds[i][1] - bounds[i][0]), self.velocity_i[i]).tolist()
    
    # Update the particle position based off new velocity updates
    def update_position(self, bounds):
        for i in range(0, num_dimensions):
            self.position_i[i] = self.position_i[i] + self.velocity_i[i]
            # Adjust maximum position if necessary
            if self.position_i[i] > bounds[i][1]:
                self.position_i[i] = bounds[i][1]
            # Adjust minimum position if neseccary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i] = bounds[i][0]      
           
class CLPSO():
    def __init__(self, costFunc, x0, bounds, num_particles, maxiter):
        global num_dimensions
        self.num_particles = num_particles
        self.maxiter = maxiter
        num_dimensions = len(x0); self.err_best_g = 10000000; self.pos_best_g = []                   
        self.pos_best_g_record = []; self.pos_best_g_record_1 = []
        self.err_best_g_record = []
        self.global_best_all_iteration = []; self.iteration = []
        self.f_pbest = []; self.pbest_f = []; self.pbest_f_1 = []; self.Pc = []
        self.fi1 = [0]*num_dimensions; self.fi2 = [0]*num_dimensions; self.fi = [0]*num_dimensions
        self.bi1 = 0; self.bi = [0]*num_dimensions; self.mintSinceLastChange = [0]*num_particles
        self.mintNuC = 5; self.mdblI = 0
        
        # Learning probability Pc
        t = np.linspace(0, 5, self.num_particles)
        self.swarm = []
        for i in range(0, num_particles):
            self.swarm.append(Particle())   
            self.Pc.append(0 + 0.5*(np.exp(t[i]) - np.exp(t[0])) / (np.exp(5) - np.exp(t[0])))
            self.f_pbest.append([i]*num_dimensions)
        
        # Initially evaluate fitness function
        for k in range(0, self.num_particles):
            self.swarm[k].evaluate(costFunc)
            
            if self.swarm[k].err_i < self.err_best_g:  
                self.pos_best_g = list(self.swarm[k].position_i)
                self.err_best_g = float(self.swarm[k].err_i)
            self.pos_best_g_record.append(self.swarm[k].pos_best_i)
            self.err_best_g_record.append(self.swarm[k].err_best_i)
            self.pbest_f.append(self.swarm[k].position_i)
            
    def Comprehensive_learning(self, num_particles, Pc, err_best_g_record, f_pbest):
        # Generate exemplar for each dimension
        for v in range(0, num_particles):
            if self.mintSinceLastChange[v] > self.mintNuC:
               self.mintSinceLastChange[v] = 0
               for z in range(0, num_dimensions): 
                  self.fi1[z] = math.ceil(random.uniform(0,1)*(num_particles-1))
                  self.fi2[z] = math.ceil(random.uniform(0,1)*(num_particles-1))
                  self.fi[z] = np.where(err_best_g_record[self.fi1[z]] < err_best_g_record[self.fi2[z]], self.fi1[z], self.fi2[z]).tolist()
                  self.bi1 = random.random() - 1 + Pc[v]
                  self.bi[z] = np.where(self.bi1 >= 0, 1, 0).tolist()   
               if np.sum(self.bi) == 0:
                  rc = round(random.uniform(0,1)*(num_dimensions - 1))
                  self.bi[rc] = 1
               for m in range(0,num_dimensions):
                  f_pbest[v][m] = self.bi[m]*self.fi[m] + (1 - self.bi[m])*f_pbest[v][m]
        return f_pbest
        
    def Run(self, costFunc):  
        i=0
        while i < self.maxiter:
            self.iteration.append(i)
            self.pos_best_g_record_1 = np.copy(self.pos_best_g_record)
            self.pbest_f_1 = np.copy(self.pbest_f)
            
            # Perform comprehensive learning strategy
            self.f_pbest = self.Comprehensive_learning(self.num_particles, self.Pc, self.err_best_g_record, self.f_pbest)
            
            # Learning from exemplars
            for j in range(0, self.num_particles):
                 for k in range(0, num_dimensions):
                         index_1 = self.f_pbest[j][k]
                         self.pbest_f_1[j,k] = self.pos_best_g_record_1[index_1, k]
            self.pbest_f = self.pbest_f_1.tolist()
            
            # Cycle through swarm and update velocities and position
            self.mdblI = 0.9 - (0.9 - 0.4) * i / self.maxiter
            for j in range(0, self.num_particles):
                self.swarm[j].update_velocity(self.pbest_f[j], self.pos_best_g, bounds, self.mdblI)
                self.swarm[j].update_position(bounds)
                
            # Cycle through particles in swarm and evaluate fitness
            for j in range(0, self.num_particles):
                self.swarm[j].evaluate(costFunc)
                # Update the personal best position and fitness values for population
                if self.swarm[j].err_i < self.err_best_g_record[j]:
                    self.pos_best_g_record[j] = list(self.swarm[j].position_i)
                    self.err_best_g_record[j] = float(self.swarm[j].err_i)
                else:
                    self.mintSinceLastChange[j] += 1
                # Determine if current particle is the best (globally)
                if self.swarm[j].err_i < self.err_best_g: 
                    self.pos_best_g = list(self.swarm[j].position_i)
                    self.err_best_g = float(self.swarm[j].err_i)
            self.global_best_all_iteration.append(self.err_best_g)
            i+=1
        # Final results
        print('FINAL RESULTS:', self.pos_best_g, '---', self.err_best_g)
        
initial = [-1, 1]   # Initial starting location [x1,x2...]
bounds = [(-2.08, 2.08),(-2.08, 2.08)] 
t = CLPSO(func1, initial, bounds, num_particles = 10, maxiter = 700)
t.Run(func1)

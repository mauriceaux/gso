#!/usr/bin/python
# encoding=utf8
import random
import reparastrategy as _repara
class SuperSwarm:
    def __init__(self,num_dimensions,min, max):
        random.seed(0)
        
        self.w=0.5       # constant inertia weight (how much to weigh the previous velocity)
        self.c3 = 2.05*random.random()        # cognative constant
        self.c4 = 2.05*random.random()        # social constant
        
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.personal_best_particle=[] # Personal best of particle
        self.member_superswarm =[] #Member i of superswarm
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual
        
        self.position_b=[]          # add particle position binario
        self.pos_best_b=[]          # add best position individual binario
        
        
        self.num_dimensions = num_dimensions
        for i in range(0,self.num_dimensions):
            self.velocity_i.append(random.uniform(-1,1))
            self.position_i.append(random.uniform(min,max))
            self.member_superswarm.append(random.uniform(min,max))
    
    def evaluate(self,costFunc, matrizCosto, matrizRestriccion,r,c):        
        self.err_i=costFunc(self.position_b, matrizCosto)
        # check to see if the current position is an individual best
        if (self.err_i < self.err_best_i or self.err_best_i==-1):
            self.personal_best_particle=self.position_i
            self.pos_best_b=self.position_b
            self.err_best_i=self.err_i            
        
        
    #inicializa enjambre del segundo nivel
    def superswarm(self,glo_best):
        for i in range(0,self.num_dimensions):
            self.velocity_i.append(random.uniform(-1,1))
        self.member_superswarm=glo_best
    
    # update new particle velocity
    def update_velocity_level_two(self,pos_best_g):
        for i in range(0,self.num_dimensions):
            r4=random.uniform(-1,1)
            r3=random.uniform(-1,1)
            vel_cognitive=self.c3*r3*(self.personal_best_particle[i]-self.member_superswarm[i])
            vel_social=self.c4*r4*(pos_best_g[i]-self.member_superswarm[i])
            self.velocity_i[i]=self.w*self.velocity_i[i]+vel_cognitive+vel_social
    
    def update_position_level_two(self,min,max):
        for i in range(0,self.num_dimensions):
            self.member_superswarm[i]=self.member_superswarm[i]+self.velocity_i[i]
            
            # adjust maximum position if necessary
            if self.member_superswarm[i]>max:
                self.member_superswarm[i]=max
    
            # adjust minimum position if neseccary
            if self.member_superswarm[i] < min:
                self.member_superswarm[i]=min 
#!/usr/bin/python
# encoding=utf8
import particle as _particle
import superswarm as _superswarm
import binarizationstrategy as _binarization
import reparastrategy as _repara
import time
import constant as _cont
import multiprocessing as mp
constantes = _cont.Const();
#import random
import numpy as np
class GSO():
    def __init__(self,costFunc, D, bounds, mcostos, mrestriccion,rutefileexecution,rutefinalfile,ejec,vrows,vcolumns,tTransferencia,tBinary):
#        random.seed(0)
        self.costFunc = costFunc
        self.vcolumns = vcolumns
        self.vrows = vrows
        self.tTransferencia = tTransferencia
        self.tBinary = tBinary
        self.mcostos = mcostos
        self.mrestriccion = mrestriccion
        time_ejecucion = time.time() #Iniciamos variable para registrar tiempo de la ejecucion
        #%--D--%%----EPmax-----%%----L1-------%%-----L2------%%---N---%%---M--%%        
        '''print('D: ' + str(D))
        print('EPMax: ' + str(constantes.EPOCH_NUMBER()))
        print('L1: ' + str(constantes.ITERATION_1()))
        print('L2: ' + str(constantes.ITERATION_2()))
        print('N: ' + str(constantes.POP_SIZE()))
        print('M: ' + str(constantes.SUB_POP()))
        print('MIN: ' + str(constantes.X_MIN()))
        print('MAX: ' + str(constantes.X_MAX()))'''
        #print(mcostos)
        
        self.err_best_g=-1 # best error for group N
        self.err_global_best_g=[] # best error for group M
        self.err_global_best=-1 # best error
        
        self.global_best_solution_subswarm=[None] * (constantes.SUB_POP())# Global best solution of subswarm i
        self.global_best_solution=[]#Global best solution for the entire swarm X
        
        self.glo_best_b=[] # global best position binario
        numT = 5
        
        # establish the swarm
        swarm=[[]] * (constantes.SUB_POP())
#        self.err_best_g=-1
        self.err_global_best = -1
        for i in range(0,constantes.SUB_POP()):
#            for d in range(D):
            if self.global_best_solution_subswarm[i] is None:
                self.global_best_solution_subswarm[i] = []
#                self.global_best_solution_subswarm[i].append(random.uniform(constantes.X_MIN(),constantes.X_MAX()))            
            args = []
            
            for j in range(constantes.POP_SIZE()):
                args.append([D,i])
            pool = mp.Pool(numT)
            swarm[i] = pool.starmap(self.createParticle, args)
            
            self.err_global_best_g.append(-1)
            
        for i in range(constantes.SUB_POP()):
            self.err_best_g=-1
            self.err_global_best_g.append(-1)
            self.err_global_best = -1
            for j in range(constantes.POP_SIZE()):
                if swarm[i][j].err_i < self.err_best_g or self.err_best_g == -1:
                    swarm[i][j].personal_best_particle = list(swarm[i][j].position_i)
                    swarm[i][j].err_best_i = float(swarm[i][j].err_i)
                    self.err_best_g=float(swarm[i][j].err_i)
                    if swarm[i][j].err_i < self.err_global_best_g[i] or self.err_global_best_g[i] == -1:
                        self.global_best_solution_subswarm[i]=list(swarm[i][j].position_i)
                        self.err_global_best_g[i]=float(swarm[i][j].err_i)
                        if swarm[i][j].err_i < self.err_global_best  or self.err_global_best == -1:
#                            print(f'i {i} {self.err_global_best_g[i]} '+str(swarm[i][j].err_i)+'<'+str(self.err_global_best_g[i]) + '-gi-best: ' + str(self.err_global_best))
#                            exit()
                            self.global_best_solution=list(swarm[i][j].position_i)
                            self.glo_best_b=list(swarm[i][j].position_b)
                            self.err_global_best=float(swarm[i][j].err_i)
                            
#        print(f'self.global_best_solution_subswarm {self.global_best_solution_subswarm}')
#        exit()
        for i in range(0,constantes.SUB_POP()):
            args.append([D,i])
        pool = mp.Pool(numT)
        
        sswarm = pool.starmap(self.createSSwarm, args)
        pool.close()
#        self.global_best_solution=list(self.global_best_solution_subswarm[1])
#        self.err_global_best=self.err_global_best_g[1]
        self.global_best_solution=list(self.global_best_solution_subswarm[np.argmax(self.err_global_best_g)])
        self.err_global_best=self.err_global_best_g[np.argmax(self.err_global_best_g)]
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #%Initialization of position and velocity vector ends
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                    
        for ep in range(0,constantes.EPOCH_NUMBER()):
            #BEGION PSO LEVEL 1
            buffer = ""
            
#            for k in range(0,constantes.ITERATION_1()):
            for i in range(0,constantes.SUB_POP()):
            
                star_time = time.time() #Iniciamos variable para registrar tiempo en cada iteración
                
#                for i in range(0,constantes.SUB_POP()):
                for k in range(0,constantes.ITERATION_1()):
                    args = []
                    for j in range(0,constantes.POP_SIZE()):
                        args.append([swarm[i][j], i, j, k])
                    
                    pool = mp.Pool(numT)
                    swarm[i] = pool.starmap(self.updateParticle, args)
                    pool.close()
                
#                for k in range(0,constantes.ITERATION_1()):
                    for j in range(0,constantes.POP_SIZE()):
                        if swarm[i][j].err_i < self.err_best_g or self.err_best_g == -1:
                            swarm[i][j].personal_best_particle = list(swarm[i][j].position_i)
                            swarm[i][j].err_best_i = float(swarm[i][j].err_i)
                            self.err_best_g=float(swarm[i][j].err_i)
    #                            
                            if self.err_best_g < self.err_global_best_g[i] or self.err_global_best_g[i] == -1:
                                self.global_best_solution_subswarm[i]=swarm[i][j].position_i
                                self.err_global_best_g[i]=float(swarm[i][j].err_i)
    #                                
                                if self.err_global_best_g[i] < self.err_global_best  or self.err_global_best == -1:
    #                                    print(f'i {i} {self.err_global_best_g[i]} '+str(swarm[i][j].err_i)+'<'+str(self.err_global_best_g[i]) + '-gi-best: ' + str(self.err_global_best))
    #                                    exit()
                                    self.global_best_solution=list(swarm[i][j].position_i)
                                    self.glo_best_b=list(swarm[i][j].position_b)
                                    self.err_global_best=float(swarm[i][j].err_i)
                tiempototaliteracion = time.time() - star_time
                buffer += 'EPMax,'+str(ep)+',M,'+str(i)+',L1,-,'+str(self.err_global_best) + ',T,' + str(tiempototaliteracion) + '\n'
            fr = open(rutefileexecution, "a+")
            fr.write(buffer)
            fr.close()
                
            # establish the swarm
            
            sswarm=[]
            args = []
            
            
            for i in range(0,constantes.SUB_POP()):
                args.append([D,i])
            pool = mp.Pool(numT)
            sswarm = pool.starmap(self.createSSwarm, args)
            pool.close()
            
            buffer = ""    
            sswarm[0].evaluate(self.costFunc, self.mcostos, self.mrestriccion,1,2)
            for k in range(0,constantes.ITERATION_2()):
                pool = mp.Pool(numT)
                star_time = time.time() #Iniciamos variable para registrar tiempo en cada iteración
                args = []
                for i in range(0,constantes.SUB_POP()):
                    args.append([sswarm[i], i, k])
                
                    
                sswarm = pool.starmap(self.updateSSwarmParticle, args)
                pool.close()
                for i in range(0,constantes.SUB_POP()):
                    if sswarm[i].err_i < self.err_global_best_g[i]  or self.err_global_best_g[i] == -1:
#                        print(str(sswarm[i].err_i)+'<'+str(self.err_global_best_g[i]) + '-gi-best: ' + str(self.err_global_best))
                        self.global_best_solution_subswarm[i]=list(sswarm[i].member_superswarm)
                        self.err_global_best_g[i]=float(sswarm[i].err_i)
                        if sswarm[i].err_i < self.err_global_best  or self.err_global_best == -1:
#                            print(str(sswarm[i].err_i)+'<'+str(self.err_global_best) + '-g-best: ' + str(self.err_global_best))
#                            exit()
                            self.global_best_solution=list(sswarm[i].member_superswarm)
                            self.glo_best_b=list(sswarm[i].position_b)
                            self.err_global_best=float(sswarm[i].err_i)          
                                
                tiempototaliteracion = time.time() - star_time
                buffer+='EPMax,'+str(ep)+',M,-,L2,'+str(k)+','+str(self.err_global_best) + ',T,' + str(tiempototaliteracion) + '\n'
#               
            fr = open(rutefileexecution, "a+")
            fr.write(buffer)
            fr.close()
            print(str(sswarm[i].err_i)+'<'+str(self.err_global_best) + '-g-best: ' + str(self.err_global_best))
#            exit()
        tt_ejecucion = time.time() - time_ejecucion #tiempo total de la ejecucion
        ff = open(rutefinalfile, "a+")
        ff.write('ejecucion;'+str(ejec)+';self.err_global_best;'+str(self.err_global_best)+';t;'+str(tt_ejecucion)+';self.glo_best_b;'+str(self.glo_best_b))
        ff.write("\n")
        ff.close()
        
    def updateSSwarmParticle(self, ss, i, iterNum):
        ss.w = 1-(iterNum/(constantes.ITERATION_2()+1))
        ss.update_velocity_level_two(self.global_best_solution)
        ss.update_position_level_two(constantes.X_MIN(),constantes.X_MAX())
        b = _binarization.BinarizationStrategy(ss.member_superswarm,self.tTransferencia,self.tBinary)
        ss.position_b = self.repara(b.get_binary())
        ss.evaluate(self.costFunc,self.mcostos,self.mrestriccion,self.vrows,self.vcolumns)
        return ss
    
    def updateParticle(self, p, subSwarmIdx, pNum, iterNum):
        p.w = 1-(iterNum/(constantes.ITERATION_1()+1))
        p.update_velocity_level_one(self.global_best_solution_subswarm[subSwarmIdx])
        p.update_position_level_one(constantes.X_MIN(),constantes.X_MAX())
        b = _binarization.BinarizationStrategy(p.position_i,self.tTransferencia,self.tBinary)
        p.position_b = self.repara(b.get_binary())
        p.evaluate(self.costFunc,self.mcostos,self.mrestriccion,self.vrows,self.vcolumns)
        return p
    
    def createParticle(self, D, i):
        p = _particle.Particle(D,constantes.X_MIN(),constantes.X_MAX())
        b = _binarization.BinarizationStrategy(p.position_i,self.tTransferencia,self.tBinary)
        p.position_b = self.repara(b.get_binary())
        p.evaluate(self.costFunc,self.mcostos,self.mrestriccion,self.vrows,self.vcolumns)
        return p
    
    def createSSwarm(self, D, i):
        ss = _superswarm.SuperSwarm(D,constantes.X_MIN(),constantes.X_MAX())
        ss.member_superswarm = self.global_best_solution_subswarm[i]
        ss.personal_best_particle = self.global_best_solution_subswarm[i]
        b = _binarization.BinarizationStrategy(ss.member_superswarm,self.tTransferencia,self.tBinary)
        ss.position_b = self.repara(b.get_binary())
        ss.evaluate(self.costFunc,self.mcostos,self.mrestriccion,self.vrows,self.vcolumns)
        return ss
    
    # Verificar Restricciones
    def repara(self,x):
        cumpleTodas=0
        repair = _repara.ReparaStrategy()
        cumpleTodas=repair.cumple(x,self.mrestriccion,self.vrows,self.vcolumns)
        if cumpleTodas==0:
            x = repair.repara_one(x,self.mrestriccion,self.mcostos,self.vrows,self.vcolumns)
        cumpleTodas = repair.cumple(x,self.mrestriccion,self.vrows,self.vcolumns)
        if cumpleTodas==0:
            x = repair.repara_two(x,self.mrestriccion,self.vrows,self.vcolumns)
        return x
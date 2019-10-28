#!/usr/bin/python
# encoding=utf8
import particle as _particle
import superswarm as _superswarm
import binarizationstrategy as _binarization
import reparastrategy as _repara
import time
import constant as _cont
import numpy as np
#import line_profiler
constantes = _cont.Const();
class GSO():
#    @profile
    def __init__(self,costFunc, D, bounds, mcostos, mrestriccion,rutefileexecution,rutefinalfile,ejec,vrows,vcolumns,tTransferencia,tBinary):
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
        err_best_g=-1 # best error for group N
        err_global_best_g=[] # best error for group M
        err_global_best=-1 # best error
        
        global_best_solution_subswarm=[]# Global best solution of subswarm i
        member_superswarm=[] #Member i of superswarm
        global_best_solution=[]#Global best solution for the entire swarm X
        
        glo_best_b=[] # global best position binario
        
        # establish the swarm
        swarm=[]
        subpop = []
        for i in range(constantes.SUB_POP()):
            swarm.append([])
            global_best_solution_subswarm.append([])
            
            err_best_g=-1
            err_global_best_g.append(-1)
            err_global_best = -1
            
            for j in range(constantes.POP_SIZE()):
                subpop.append(_particle.Particle(D,constantes.X_MIN(),constantes.X_MAX()))
                swarm[i].append(subpop[j])        
#                print(type(swarm[i][j].position_i[0]))
#                exit()
                b = _binarization.BinarizationStrategy(swarm[i][j].position_i,tTransferencia,tBinary)
                #print(b.get_binary(),mcostos,mrestriccion,vrows,vcolumns)
                swarm[i][j].position_b = self.repara(b.get_binary(),mcostos,mrestriccion,vrows,vcolumns)
                #print(swarm[i][j].position_b)
                #swarm[i][j].binariza(tTransferencia,tBinary,1)
                swarm[i][j].evaluate(costFunc,mcostos,mrestriccion,vrows,vcolumns)
                if swarm[i][j].err_i < err_best_g or err_best_g == -1:
                    swarm[i][j].personal_best_particle = list(swarm[i][j].position_i)
                    swarm[i][j].err_best_i = float(swarm[i][j].err_i)
                    err_best_g=float(swarm[i][j].err_i)
#                    print(str(swarm[i][j].err_i)+'<'+str(err_best_g) + '-err: ' + str(err_global_best))
#                    exit()
                    if swarm[i][j].err_i < err_global_best_g[i] or err_global_best_g[i] == -1:
                        global_best_solution_subswarm[i]=list(swarm[i][j].position_i)
                        err_global_best_g[i]=float(swarm[i][j].err_i)
                        #print(str(swarm[i][j].err_i)+'<'+str(err_global_best_g[i]) + '-err: ' + str(err_global_best))
                        if swarm[i][j].err_i < err_global_best  or err_global_best == -1:
                            global_best_solution=list(swarm[i][j].position_i)
                            glo_best_b=list(swarm[i][j].position_b)
                            err_global_best=float(swarm[i][j].err_i)
#                            print(str(swarm[i][j].err_i)+'<'+str(err_global_best) + '-err: ' + str(err_global_best))
        
        sswarm=[]
        for i in range(0,constantes.SUB_POP()):
            member_superswarm.append([])
            sswarm.append(_superswarm.SuperSwarm(D,constantes.X_MIN(),constantes.X_MAX()))
                
        global_best_solution=list(global_best_solution_subswarm[1])
        err_global_best=err_global_best_g[1]
        
        for i in range(0,constantes.SUB_POP()):
            b = _binarization.BinarizationStrategy(sswarm[i].member_superswarm,tTransferencia,tBinary)
            sswarm[i].position_b = self.repara(b.get_binary(),mcostos,mrestriccion,vrows,vcolumns)
            #sswarm[i].binariza(tTransferencia,tBinary,2)
            sswarm[i].evaluate(costFunc,mcostos,mrestriccion,vrows,vcolumns)
            if sswarm[i].err_i < err_global_best_g[i]  or err_global_best_g[i] == -1:
                global_best_solution_subswarm[i]=list(sswarm[i].member_superswarm)
                err_global_best_g[i]=float(sswarm[i].err_i)
                #print(str(sswarm[i].err_i)+'<'+str(err_global_best_g[i]) + '-err: ' + str(err_global_best))
                if sswarm[i].err_i < err_global_best  or err_global_best == -1:
                        global_best_solution=list(sswarm[i].position_i)
                        glo_best_b=list(sswarm[i].position_b)
                        err_global_best=float(sswarm[i].err_i)        
                        #print(str(sswarm[i].err_i)+'<'+str(err_global_best) + '-err: ' + str(err_global_best))

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #%Initialization of position and velocity vector ends
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                    
        for ep in range(0,constantes.EPOCH_NUMBER()):
            #BEGION PSO LEVEL 1
            
            for i in range(0,constantes.SUB_POP()):
                evaluationsCsv = []
                star_time = time.time() #Iniciamos variable para registrar tiempo en cada iteración
                for k in range(0,constantes.ITERATION_1()):                    
                    evals = []
                    for j in range(0,constantes.POP_SIZE()):
                        
                        swarm[i][j].update_velocity_level_one(global_best_solution_subswarm[i])
                        swarm[i][j].update_position_level_one(constantes.X_MIN(),constantes.X_MAX())
                        b = _binarization.BinarizationStrategy(swarm[i][j].position_i,tTransferencia,tBinary)
                        swarm[i][j].position_b = self.repara(b.get_binary(),mcostos,mrestriccion,vrows,vcolumns)
                        #swarm[i][j].binariza(tTransferencia,tBinary,1)
                        swarm[i][j].evaluate(costFunc,mcostos,mrestriccion,vrows,vcolumns)
                        evals.append(swarm[i][j].err_i)
                        if swarm[i][j].err_i < err_best_g or err_best_g == -1:
                            #print(str(swarm[i][j].err_i)+'<'+str(err_best_g) + '-pb-best: ' + str(err_global_best))
                            swarm[i][j].personal_best_particle = list(swarm[i][j].position_i)
                            swarm[i][j].err_best_i = float(swarm[i][j].err_i)
                            err_best_g=float(swarm[i][j].err_i)
                            
                            print(f'err_best_g {err_best_g}')
                            if err_best_g < err_global_best_g[i] or err_global_best_g[i] == -1:
                                #print(str(swarm[i][j].err_i)+'<'+str(err_global_best_g[i]) + '-gi-best: ' + str(err_global_best))
                                global_best_solution_subswarm[i]=list(swarm[i][j].position_i)
                                err_global_best_g[i]=float(swarm[i][j].err_i)
                                
                                if err_global_best_g[i] < err_global_best  or err_global_best == -1:
#                                    print(str(swarm[i][j].err_i)+'<'+str(err_global_best) + '-g-best: ' + str(err_global_best))
#                                    exit()
                                    global_best_solution=list(swarm[i][j].position_i)
                                    glo_best_b=list(swarm[i][j].position_b)
                                    err_global_best=float(swarm[i][j].err_i)
                    evaluationsCsv.append(evals)
                                    
                tiempototaliteracion = time.time() - star_time
                fr = open(rutefileexecution, "a+")
                fr.write('EPMax,'+str(ep)+',M,'+str(i)+',L1,-,'+str(err_global_best) + ',T,' + str(tiempototaliteracion))
                fr.write("\n")
                fr.close()
                np.savetxt(f"resultados/swarmL{0}S{i}.csv", np.array(evaluationsCsv), delimiter=",")
                #print('EPMax,'+str(ep)+',M,'+str(i)+',L1,-,'+str(err_global_best) + ',T,' + str(tiempototaliteracion))
                
            # establish the swarm
            for i in range(0,constantes.SUB_POP()): 
                sswarm[i].superswarm(global_best_solution_subswarm[i])
            
            #BEGIN PSO LEVEL 2 
            evaluationsCsv = []                       
            for k in range(0,constantes.ITERATION_2()):
                evals = []
                star_time = time.time() #Iniciamos variable para registrar tiempo en cada iteración
                for i in range(0,constantes.SUB_POP()):
                    sswarm[i].update_velocity_level_two(global_best_solution)
                    sswarm[i].update_position_level_two(constantes.X_MIN(),constantes.X_MAX())
#                    import numpy as np
#                    print(f'sswarm[i].member_superswarm {np.array(sswarm[i].member_superswarm).shape}')
#                    exit()
                    b = _binarization.BinarizationStrategy(sswarm[i].member_superswarm,tTransferencia,tBinary)
                    sswarm[i].position_b = self.repara(b.get_binary(),mcostos,mrestriccion,vrows,vcolumns)
                    #sswarm[i].binariza(tTransferencia,tBinary,2)
                    sswarm[i].evaluate(costFunc,mcostos,mrestriccion,vrows,vcolumns)
                    evals.append(sswarm[i].err_i)
                    # determine if current particle is the best (globally)
                    if sswarm[i].err_i < err_global_best_g[i]  or err_global_best_g[i] == -1:
                        #print(str(sswarm[i].err_i)+'<'+str(err_global_best_g[i]) + '-gi-best: ' + str(err_global_best))
                        global_best_solution_subswarm[i]=list(sswarm[i].member_superswarm)
                        err_global_best_g[i]=float(sswarm[i].err_i)
                        
                        if sswarm[i].err_i < err_global_best  or err_global_best == -1:
#                            print(str(sswarm[i].err_i)+'<'+str(err_global_best) + '-g-best: ' + str(err_global_best))
#                            exit()
                            global_best_solution=list(sswarm[i].member_superswarm)
                            glo_best_b=list(sswarm[i].position_b)
                            err_global_best=float(sswarm[i].err_i)          
                evaluationsCsv.append(evals)                
                tiempototaliteracion = time.time() - star_time
                fr = open(rutefileexecution, "a+")
                fr.write('EPMax,'+str(ep)+',M,-,L2,'+str(k)+','+str(err_global_best) + ',T,' + str(tiempototaliteracion))
                fr.write("\n")
                fr.close()
                #print ('EPMax,'+str(ep)+',M,-,L2,'+str(k)+','+str(err_global_best) + ',T,' + str(tiempototaliteracion))
            print(str(sswarm[i].err_i)+'<'+str(err_global_best) + '-g-best: ' + str(err_global_best))
            np.savetxt(f"resultados/swarmL{1}S{0}.csv", np.array(evaluationsCsv), delimiter=",")
#            exit()
        tt_ejecucion = time.time() - time_ejecucion #tiempo total de la ejecucion
        ff = open(rutefinalfile, "a+")
        ff.write('ejecucion;'+str(ejec)+';err_global_best;'+str(err_global_best)+';t;'+str(tt_ejecucion)+';glo_best_b;'+str(glo_best_b))
        ff.write("\n")
        ff.close()
    
    # Verificar Restricciones
#    @profile
    def repara(self,x, matrizCosto, matrizRestriccion,r,c):
        cumpleTodas=0
        repair = _repara.ReparaStrategy()
        cumpleTodas=repair.cumple(x,matrizRestriccion,r,c)
        if cumpleTodas==0:
            x = repair.repara_one(x,matrizRestriccion,matrizCosto,r,c)    
        cumpleTodas = repair.cumple(x,matrizRestriccion,r,c)
        if cumpleTodas==0:
            x = repair.repara_two(x,matrizRestriccion,r,c)    
        return x
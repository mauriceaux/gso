import itertools 
import json

instancias = ["mscp42.txt"
                ,"mscp52.txt"
                ,"mscp62.txt"
                ,"mscpa2.txt"
                ,"mscpb2.txt"
                ,"mscpc2.txt"
                ,"mscpd2.txt"
                ,"mscpnre2.txt"
                ,"mscpnrf2.txt"
                ,"mscpnrg2.txt"
                ,"mscpnrh2.txt"
                ]

parametros = ["inercia"
                ,"accelPer"
                ,"accelBest"
                ,"numParticulas"]

permParam = []
for i in range(len(parametros)):
    if i == 0: continue
    c = list(itertools.combinations(parametros,i))
    unq = list(set(c))
    permParam.extend(unq)

permParam.append(["inercia"
                ,"accelPer"
                ,"accelBest"
                ,"numParticulas"])

with open("insert.sql", "a") as file_object:
    for instancia in instancias:
        for perm in permParam:
            for i in range(31):
                parametros = {
                    "instancia" : instancia
                    ,"paramOptimizar" : perm
                }
                query = f"INSERT INTO datos_ejecucion (nombre_algoritmo, parametros, estado) VALUES ('GSO', '{json.dumps(parametros)}', 'pendiente');\n"
                file_object.write(query)






cambio insignificante para el git


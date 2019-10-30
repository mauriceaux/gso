import reparastrategy as _repara
import numpy as np
from datetime import datetime

reparacion = _repara.ReparaStrategy()
m_restriccion= np.array([[1,0,0,1],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
costos = np.array([2,1,3,4])
reparacion.m_restriccion = m_restriccion
reparacion.m_costos = costos
reparacion.genImpRestr()

solucion = [0,0,0,0]
start = datetime.now()
sol1 = reparacion.repara_oneModificado(solucion, m_restriccion,costos,4,4)
end = datetime.now()
print(f'repara modificado demoro {end-start}')
#start = datetime.now()
#sol2 = reparacion.repara_one([0,0,0,0], m_restriccion,costos,4,4)
#end = datetime.now()
#print(f'repara original demoro {end-start}')

print(f'sol1 {sol1}')
#print(f'sol2 {sol2}')
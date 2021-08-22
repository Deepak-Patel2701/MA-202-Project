
#2) Plot of falling time vs time
import numpy as np
from matplotlib import pyplot as plt

g = 9.8
eta =  1.85*(10**(-5))
r = 5*(10**(-8))
p_p = 1.35*(10**(3))

m = p_p*4*np.pi*(r**(3))/3 

def dv_dt(v):
  return (g - ((6*np.pi*r*eta)*v)/m)

day = 24*3600
temp = 0
time = []
delta_t = 0.00001/day
while(temp<=0.1/day):
  time.append(temp)
  temp+=delta_t

v = [0]*len(time)
timeArray = [0]*len(time)
timeArray[0]=2/5
v[0] = 5

plt.figure(figsize=(8,6), dpi = 100)
for i in range (len(time)-1):
  v[i+1] = v[i] + (delta_t*dv_dt(v[i]))
  timeArray[i+1] = 0.3/v[i+1] 
plt.plot(time,timeArray, label = 'Height = 0.3')

for i in range (len(time)-1):
  v[i+1] = v[i] + (delta_t*dv_dt(v[i]))
  timeArray[i+1] = 0.6/v[i+1] 
plt.plot(time,timeArray, label = 'Height = 0.6')

for i in range (len(time)-1):
  v[i+1] = v[i] + (delta_t*dv_dt(v[i]))
  timeArray[i+1] = 0.9/v[i+1] 
plt.plot(time,timeArray, label = 'Height = 0.9')

for i in range (len(time)-1):
  v[i+1] = v[i] + (delta_t*dv_dt(v[i]))
  timeArray[i+1] = 1.2/v[i+1] 
plt.plot(time,timeArray, label = 'Height = 1.2')

for i in range (len(time)-1):
  v[i+1] = v[i] + (delta_t*dv_dt(v[i]))
  timeArray[i+1] = 1.5/v[i+1] 
plt.plot(time,timeArray, label = 'Height = 1.5')

for i in range (len(time)-1):
  v[i+1] = v[i] + (delta_t*dv_dt(v[i]))
  timeArray[i+1] = 1.8/v[i+1] 
plt.plot(time,timeArray, label = 'Height = 1.8')

for i in range (len(time)-1):
  v[i+1] = v[i] + (delta_t*dv_dt(v[i]))
  timeArray[i+1] = 2.1/v[i+1] 
plt.plot(time,timeArray, label = 'Height = 2.1')


plt.plot(time,timeArray)
plt.title('Finding V_terminal for different Heights | Using Stokes Law')
plt.xlabel('Time in [s]')
plt.ylabel('Falling time in [s]')
plt.grid(True) 
plt.legend(loc='best')
plt.show()





###############################


# 2) Plot of falling time vs time required to attain v_terminal

import numpy as np
from matplotlib import pyplot as plt

g = 9.8
k = 0.47
p_a = 1.2041
p_p = 1.35*(10**3)
r = 5*(10**(-8))

def dv_dt(v):
  return (g - (3/8*k*p_a*(v**2)/p_p/r))

temp = 0
time = []
delta_t = 0.00001
while(temp<=0.1):
  time.append(temp)
  temp+=delta_t

#print(time)
v = [0]*len(time)
timeArray = [0]*len(time)
timeArray[0]=2/5
v[0] = 5
plt.figure(figsize=(8,6), dpi = 100)
for i in range (len(time)-1):
  v[i+1] = v[i] + (delta_t*dv_dt(v[i]))
  timeArray[i+1] = 0.3/v[i+1] 
plt.plot(time,timeArray, label = 'Height = 0.3')

for i in range (len(time)-1):
  v[i+1] = v[i] + (delta_t*dv_dt(v[i]))
  timeArray[i+1] = 0.6/v[i+1] 
plt.plot(time,timeArray, label = 'Height = 0.6')

for i in range (len(time)-1):
  v[i+1] = v[i] + (delta_t*dv_dt(v[i]))
  timeArray[i+1] = 0.9/v[i+1] 
plt.plot(time,timeArray, label = 'Height = 0.9')

for i in range (len(time)-1):
  v[i+1] = v[i] + (delta_t*dv_dt(v[i]))
  timeArray[i+1] = 1.2/v[i+1] 
plt.plot(time,timeArray, label = 'Height = 1.2')

for i in range (len(time)-1):
  v[i+1] = v[i] + (delta_t*dv_dt(v[i]))
  timeArray[i+1] = 1.5/v[i+1] 
plt.plot(time,timeArray, label = 'Height = 1.5')

for i in range (len(time)-1):
  v[i+1] = v[i] + (delta_t*dv_dt(v[i]))
  timeArray[i+1] = 1.8/v[i+1] 
plt.plot(time,timeArray, label = 'Height = 1.8')

for i in range (len(time)-1):
  v[i+1] = v[i] + (delta_t*dv_dt(v[i]))
  timeArray[i+1] = 2.1/v[i+1] 
plt.plot(time,timeArray, label = 'Height = 2.1')


plt.plot(time,timeArray)
plt.title('Falling time vs time at instanteous velocity | Using Newtons Law')
plt.xlabel('Time in [s]')
plt.ylabel('Falling time in [s]')
plt.grid(True) 
plt.legend(loc='best')
plt.show()





#############################



import numpy as np
from matplotlib import pyplot as plt

g = 9.8
k = 0.47
p_a = 1.2041
p_p = 1.35*(10**3)
r = 5*(10**(-8))

def dv_dt(v):
  return (g - (3/8*k*p_a*(v**2)/p_p/r))

temp = 0
time = []
delta_t = 0.00001
while(temp<=0.1):
  time.append(temp)
  temp+=delta_t

v = [0]*len(time)
timeArray = [0]*len(time)
timeArray[0]=2/5
v[0] = 5

for i in range (len(time)-1):
  v[i+1] = v[i] + (delta_t*dv_dt(v[i]))
  timeArray[i+1] = 2/v[i+1] 
  
v_ter = v[len(time)-1]

height = np.arange(0,2.2,0.3)
timeArray = [0]*len(height)

for i in range(len(height)):
  timeArray[i] = height[i]/ v_ter

plt.figure(figsize=(8,6), dpi = 100)
plt.plot(timeArray, height, 'r-o') 
plt.title("Height vs Time |  Using Newton's Law")
plt.xlabel('Time in [s]')
plt.ylabel('Height of Human [m]')
plt.grid(True) 
plt.show()




##########################################



# 1) Plot of time vs v_terminal ( ie time at which we attain terminal velocity)
# 2) Plot of falling time vs time required to attain v_terminal

import numpy as np
from matplotlib import pyplot as plt

g = 9.8
eta =  1.85*(10**(-5))
r = 5*(10**(-8))
p_p = 1.35*(10**(3))

m = p_p*4*np.pi*(r**(3))/3 

def dv_dt(v):
  return (g - ((6*np.pi*r*eta)*v)/m)
day = 24*3600
temp = 0
time = []
delta_t = 0.00001/day
while(temp<=0.1/day):
  time.append(temp)
  temp+=delta_t

#print(time)
v = [0]*len(time)
v[0] = 5

plt.figure(figsize=(8,6), dpi = 100)
for i in range (len(time)-1):
  v[i+1] = v[i] + (delta_t*dv_dt(v[i]))

print('The terminal velocity (in m/s) is : ', v[len(time)-1])
plt.plot(time,v)
plt.title('Velocity of the particle vs time')
plt.xlabel('Time in sec')
plt.ylabel('Velocity in m/s')
plt.grid(True) 
# plt.legend(loc='best')
plt.show()





#####################################





#2) Plot of falling time vs time
import numpy as np
from matplotlib import pyplot as plt

g = 9.8
eta =  1.85*(10**(-5))
r = 5*(10**(-8))
p_p = 1.35*(10**(3))

m = p_p*4*np.pi*(r**(3))/3 

def dv_dt(v):
  return (g - ((6*np.pi*r*eta)*v)/m)

day = 24*3600
temp = 0
time = []
delta_t = 0.00001/day
while(temp<=0.1/day):
  time.append(temp)
  temp+=delta_t

v = [0]*len(time)
timeArray = [0]*len(time)
timeArray[0]=2/5
v[0] = 5

plt.figure(figsize=(8,6), dpi = 100)
for i in range (len(time)-1):
  v[i+1] = v[i] + (delta_t*dv_dt(v[i]))
  timeArray[i+1] = 0.3/v[i+1] 
plt.plot(time,timeArray, label = 'Height = 0.3')

for i in range (len(time)-1):
  v[i+1] = v[i] + (delta_t*dv_dt(v[i]))
  timeArray[i+1] = 0.6/v[i+1] 
plt.plot(time,timeArray, label = 'Height = 0.6')

for i in range (len(time)-1):
  v[i+1] = v[i] + (delta_t*dv_dt(v[i]))
  timeArray[i+1] = 0.9/v[i+1] 
plt.plot(time,timeArray, label = 'Height = 0.9')

for i in range (len(time)-1):
  v[i+1] = v[i] + (delta_t*dv_dt(v[i]))
  timeArray[i+1] = 1.2/v[i+1] 
plt.plot(time,timeArray, label = 'Height = 1.2')

for i in range (len(time)-1):
  v[i+1] = v[i] + (delta_t*dv_dt(v[i]))
  timeArray[i+1] = 1.5/v[i+1] 
plt.plot(time,timeArray, label = 'Height = 1.5')

for i in range (len(time)-1):
  v[i+1] = v[i] + (delta_t*dv_dt(v[i]))
  timeArray[i+1] = 1.8/v[i+1] 
plt.plot(time,timeArray, label = 'Height = 1.8')

for i in range (len(time)-1):
  v[i+1] = v[i] + (delta_t*dv_dt(v[i]))
  timeArray[i+1] = 2.1/v[i+1] 
plt.plot(time,timeArray, label = 'Height = 2.1')


plt.plot(time,timeArray)
plt.title('Finding V_terminal for different Heights | Using Stokes Law')
plt.xlabel('Time in [s]')
plt.ylabel('Falling time in [s]')
plt.grid(True) 
plt.legend(loc='best')
plt.show()








###################################






import numpy as np
from matplotlib import pyplot as plt

g = 9.8
k = 0.47
p_a = 1.2041
p_p = 1.35*(10**3)
r = 5*(10**(-8))

def dv_dt(v):
  return (g - ((6*np.pi*r*eta)*v)/m)

day = 24*3600
temp = 0
time = []
delta_t = 0.00001/day
while(temp<=0.1/day):
  time.append(temp)
  temp+=delta_t

v = [0]*len(time)
timeArray = [0]*len(time)
timeArray[0]=2/5
v[0] = 5

for i in range (len(time)-1):
  v[i+1] = v[i] + (delta_t*dv_dt(v[i]))
  timeArray[i+1] = 2/v[i+1] 
  
v_ter = v[len(time)-1]

height = np.arange(0,2.2,0.3)
timeArray = [0]*len(height)

for i in range(len(height)):
  timeArray[i] = height[i]/ v_ter


plt.figure(figsize=(8,6), dpi = 100)
plt.plot(timeArray, height, 'r-o') 
plt.title("Height vs Time |  Using Stoke's Law")
plt.xlabel('Time in [s]')
plt.ylabel('Height of Human [m]')
plt.grid(True) 
plt.show()







#####################################





# Finding horizontal distance for height = 2m

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Breething for height = 2m
v = 1.5
g = 9.8
x = np.linspace(0,10,100)
def calculate_fx(x,h):
  return (x - (v*((2*h/g)**(0.5))))

fx = [0]*len(x)
for i in range (len(x)):
  fx[i] = calculate_fx(x[i],2)
plt.figure(figsize=(10,8),dpi=100)
plt.title("Graph")
plt.ylabel("F(x)")
plt.plot(x,fx,'-', color='gray')
plt.grid("True")
plt.show()

e_s = 0.000001  #Specified tolerance
# Setting values of x_l = 2, x_u = 4 from the plot above
x_u = 2
x_l = 0
x_r = 0.5*(x_l + x_u)
dist_r = calculate_fx(x_r, 2)
dist_l = calculate_fx(x_l, 2)

if(dist_r*dist_l<0):
    #Root in lower subinterval
    x_u = x_r
elif (dist_r*dist_l>0): #root in upper subinterval
    x_l = x_r
x_new = 0.5*(x_l + x_u)
e_a = abs((x_new - x_r)/x_new)
while(e_a>e_s): #While error calculated is less than tolerance specified
    x_r = x_new
    dist_r = calculate_fx(x_r, 2)
    dist_l = calculate_fx(x_l, 2)
    if(dist_r*dist_l<0):
        #Root in lower subinterval
        x_u = x_r
    elif (dist_r*dist_l>0): #root in upper subinterval
        x_l = x_r
    elif (dist_r*dist_l == 0):
        break
    x_new = 0.5*(x_l + x_u)
    e_a = abs((x_new - x_r)/x_new)
    #print(x_l,x_r,x_u, x_new, e_a)
    #print(e_a)
print("For Height = 2m, the horizontal distance travelled by the particle e_s = 10^-6 is:", x_r)




#####################################



#Analytical sol for Newtons Law
import numpy as np
from matplotlib import pyplot as plt

g = 9.8
k = 0.47
p_a = 1.2041
p_p = 1.35*(10**3)
r = 5*(10**(-8))
A = np.pi*r*r
m = (p_p*4*np.pi*r*r*r)/3

v_ter = ((2*m*g)/(k*p_a*A))**(0.5)
h = np.linspace(0,2.1,1000,endpoint=True)

plt.plot(h,h/v_ter,label = "Falling time")
plt.grid(True)
plt.title("Analytical solution for height vs falling time")
plt.xlabel("Height in m")
plt.ylabel("Falling time in sec")







########################################




#Analytical sol for Stokes Law
import numpy as np
from matplotlib import pyplot as plt

g = 9.8
eta =  1.85*(10**(-5))
r = 5*(10**(-8))
p_p = 1.35*(10**(3))

m = p_p*4*np.pi*(r**(3))/3 

v_ter = (m*g)/(6*np.pi*eta*r)
h = np.linspace(0,2.1,1000,endpoint=True)

plt.plot(h,h/v_ter,label = "Falling time")
plt.grid(True)
plt.title("Analytical solution for height vs falling time")
plt.xlabel("Height in m")
plt.ylabel("Falling time in sec")





#########################################






#Analytical sol for Horizontal Distance Newtons Law (Sneezing) 
import numpy as np
from matplotlib import pyplot as plt

g = 9.8
eta =  1.85*(10**(-5))
r = 5*(10**(-8))
p_p = 1.35*(10**(3))

m = p_p*4*np.pi*(r**(3))/3 

x = (1.5*((2*h/g)**(0.5)))
h = np.linspace(0,2.1,1000,endpoint=True)

plt.plot(h,x)
plt.grid(True)
plt.title("Analytical solution for Horizontal Distance vs Human Height")
plt.xlabel("Height in m")
plt.ylabel("Horizontal Distance in m")







########################################





#Analytical sol for Horizontal Distance Newtons Law (Sneezing) 
import numpy as np
from matplotlib import pyplot as plt

g = 9.8
eta =  1.85*(10**(-5))
r = 5*(10**(-8))
p_p = 1.35*(10**(3))

m = p_p*4*np.pi*(r**(3))/3 


h = np.linspace(0,2.1,1000,endpoint=True)
x = (1.5*((2*h/g)**(0.5)))

plt.plot(h,x)
plt.grid(True)
plt.title("Analytical solution for Horizontal Distance vs Human Height")
plt.xlabel("Height in m")
plt.ylabel("Horizontal Distance in m")






#######################################








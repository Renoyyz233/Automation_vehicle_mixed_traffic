# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

''' Discription
% Driver Model - OVM
% Homogeneous Setting for alpha, beta and fv
% Initial velocity and position are in equilibrium
% One vehicle will Brake sharply
% Add a safe distance algorithm to guarantee safety (not crash the
% preceding vehicle), including HDV and AV
% Correspond to Fig. 12 in our paper.
'''

import math
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

# In[1] 
''' Key Parameters'''

N = 20
AV_number = 4 # 0 or 1 or 2 or 4

platoon_bool = 0

# Position of the perturbation
brakeID = 15


# In[2] 
''' Parameters '''

if AV_number == 0:
    mix = 0
    ActuationTime = 9999
else:
    mix = 1;

ID = np.zeros([N]); #0. Manually Driven  1. Controller

if mix:
    ActuationTime = 0;
    # Define the spatial formation of the AVs
    
    if AV_number == 4:
        if platoon_bool:
            ID[8] = 1
            ID[9] = 1
            ID[10] = 1
            ID[11] = 1
        else:
            ID[2] = 1
            ID[7] = 1
            ID[12] = 1
            ID[17] = 1
            
    if AV_number == 2:
        if platoon_bool:
            ID[9] = 1
            ID[10] = 1
        else:
            ID[4] = 1
            ID[14] = 1
            
    if AV_number == 1:
        ID[19] = 1
        

#Controller Parameter
gammaType = 2;

v_star = 15;

# OVM parameter
s_star = 20;
v_max  = 30;
s_st   = 5;
s_go   = 35;

'''%%%%% Type1 %%%%%%%'''
alpha  = 0.6
beta   = 0.9

'''%%%%%%%%% Type2 %%%%%%%%%'''
#     alpha  = 1.0;
#     beta   = 1.5;



# In[3] 
'''Other Parameters'''

acel_max = 2;
dcel_max = -5;

'''%Driver Model: OVM'''

'''% safe distance for collision avoidance'''
sd = 8; # minimum value is zero since the vehicle length is ignored

#Simulation
TotalTime = 100;
Tstep = 0.01;
NumStep = int(TotalTime/Tstep);
#Scenario
Circumference = s_star*N;


#Initial State for each vehicle
S = np.zeros((NumStep,N,3))
dev_s = 0;
dev_v = 0;
co_v = 1.0;
v_ini = co_v*v_star; #Initial velocity
#from -dev to dev

var1 = np.linspace(Circumference, s_star, N)
var2 = np.random.rand(N)*2*dev_s-dev_s    
S[0, :, 0] = var1 + var2
    
var1 = v_ini*np.ones([N])
var2 = (np.random.rand(N)*2*dev_v-dev_v)    
S[0, :, 1] =  var1 + var2

# In[4] 
#Velocity Difference
V_diff = np.zeros([NumStep,N]);
#Following Distance
D_diff = np.zeros([NumStep,N]);
temp = np.zeros([N]);
#Avg Speed
V_avg = np.zeros((NumStep,1));

X = np.zeros((2*N,NumStep));


# In[4] 
##Controller

alpha1 = alpha*v_max/2* math.pi /(s_go-s_st)* math.sin (math.pi*(s_star-s_st)/(s_go-s_st));
alpha2 = alpha+beta;
alpha3 = beta;


# In[function] 
def ReturnObjectiveValue(AV_ID, N, alpha1, alpha2, alpha3, gammaType) :
    # Generate the system model and the optimal objective value
    
    if(gammaType == 1) :
    #S1
        gamma_s = 0.01
        gamma_v = 0.05
        gamma_u = 0.1
        
    elif(gammaType == 2) :
    #S2
        gamma_s = 0.03
        gamma_v = 0.15
        gamma_u = 0.1
        
    elif(gammaType == 3) :
    #S3
        gamma_s = 0.05
        gamma_v = 0.25
        gamma_u = 0.1
         
    elif(gammaType == 4) :
    #S4
        gamma_s = 0.03
        gamma_v = 0.15
        gamma_u = 1
    
    elif(gammaType == 5) :
        gamma_s = 1
        gamma_v = 1
        gamma_u = 0
        
    elif(gammaType == 9999) :
        gamma_s = 0.01
        gamma_v = 0.05
        gamma_u = 1e-6

    AV_number = np.count_nonzero(AV_ID)
    A1 = [[0,-1],[alpha1,-alpha2]]
    A2 = [[0,1],[0,alpha3]]
    C1 = [[0,-1],[0,0]]
    C2 = [[0,1],[0,0]]
    
    A = np.zeros((2 * N, 2 * N))
    B = np.zeros((2 * N, AV_number))
    Q = np.zeros((2 * N, 2 * N))

    for i in range(1, N + 1) :
        Q[2 * i - 2, 2 * i - 2] = gamma_s
        Q[2 * i - 1, 2 * i - 1] = gamma_v
    
    R = gamma_u * np.eye(AV_number)

    A[0:2,0:2] = A1
    A[0:2,(2 * N - 1) - 1:2 * N] = A2

    for i in range(2, N + 1) : 
        A[(2 * i - 1) - 1 : (2 * i) , (2 * i - 1) - 1 : (2 * i)] = A1
        A[(2 * i - 1) - 1 : (2 * i) , (2 * i - 3) - 1 : (2 * i - 2)] = A2

    if (alpha2 ** 2) - (alpha3 ** 2) - (2 * alpha1) > 0 :
        stability_condition_bool = True
    else :
        stability_condition_bool = False

    temp_x = np.nonzero((np.real(np.linalg.eig(A)[0]) > 0.001)[0])
    if temp_x == 0 :
        stable_bool = True
    else :
        stable_bool = False

    k = 1
    for i in range(1, N + 1):
        if ID[i - 1] == 1:
            if i == 1:
                A[1 - 1 : 2, 1 - 1 : 2] = C1
                A[1 - 1 : 2, (2 * N - 1 - 1) : 2 * N] = C2
            else:
                A[(2 * i - 2) : (2 * i) , (2 * i - 2) : (2 * i)] = C1
                A[(2 * i - 2) : (2 * i) , (2 * i - 4) : (2 * i - 2)] = C2
            B[2 * i - 1, k - 1] = 1
            k = k + 1

    # Call Yalmip to calculate the optimum
    epsilon   = 1e-5

    n = len(A)  # number of states
    m = len(B[0]) # number of inputs

    # assume each vehicle has a deviation
    B1 = np.eye(n)
    B1[0 : n : 2, 0 : n : 2] = 0

    # B1 = B;
    # assume disturbance is the same as input

    # variables
    # X = cp.Variable(shape = (n,n))
    # Z = cp.Variable(shape = (m,n))
    # Y = cp.Variable(shape = (m,m))
    # constraints = [M + M.T + (B1 @ B1.T) <= 0,\
    #    X - epsilon *  np.eye(n) >= 0,\
    #    cp.vstack((cp.hstack((Y,Z)), cp.hstack((Z.T,X)))) >= 0]

    S = cp.Variable((m + n, m + n), symmetric = True)

    # constraints
    constraints  = [(A @ S[m:,m:] - B @ S[0 : m,m:]) + (A @ S[m:,m:] - B @ S[0 : m,m:]).T + B1 @ B1.T << 0 ,\
         S[m:,m:] - epsilon * np.eye(n) >> 0,\
         S >> 0]

    # print(cp.installed_solvers())

    obj = cp.trace(Q @ S[m:,m:]) + cp.trace(R @ S[0:m,0:m])
    problem = cp.Problem(cp.Minimize(obj), constraints)
    problem.solve(solver = cp.MOSEK, verbose = True)

    Xd = S[m:,m:].value
    Zd = S[0:m,m:].value
    Yd = S[0:m,0:m].value

    K = Zd @ np.linalg.inv(Xd)
    Obj = cp.trace(Q @ Xd) + cp.trace(R @ Yd)

    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=100)
    np.set_printoptions(precision=4)
    print(K)

    return Obj,stable_bool,stability_condition_bool,K

# In[apply f] 
if mix:
    Obj,stable_bool,stability_condition_bool,K = ReturnObjectiveValue(ID,N,alpha1,alpha2,alpha3,gammaType);


# In[5] 

#Simulation
for k in range(0,NumStep-2):
    
    #Car in front velocity
    temp[1:] = S[k,:-1,1]
    temp[0] = S[k,-1,1]
    V_diff[k,:] = temp-S[k,:,1]
    
    temp[0]=S[k,-1,0]+ Circumference
    temp[1:] = S[k,:-1,0]

    D_diff[k,:] = temp-S[k,:,0]

    cal_D = D_diff[k,:]

    ## might be different
    cal_D[cal_D>s_go] = s_go
    cal_D[cal_D<s_st] = s_st


    acel2 = math.pi*(cal_D-s_st)/(s_go-s_st)
    acel1 = (1-np.cos(acel2))
    acel3 = np.zeros(N)
    
    
    acel = alpha*(v_max/2*acel1-S[k,:,1])+beta*V_diff[k,:] + acel3
    acel[acel>acel_max] = acel_max
    acel[acel<dcel_max] = dcel_max
    
    
    #SD as ADAS to prevent crash
    temp[1:] = S[k,:-1,1]
    temp[0] = S[k,-1,1]
    acel_sd = (S[k,:,1]**2-temp**2)/2/(D_diff[k,:]-sd)
    #if (k%100==0):
        #print(D_diff[k,:])
        #print(temp)
        #print(acel_sd)
        #print(" ")
    acel[acel_sd>abs(dcel_max)] = dcel_max
    
    S[k,:,2] = acel

    if mix:
        AV_position = np.nonzero(ID == 1)
        if (k> ActuationTime/Tstep):
            X[np.arange(0,2*N,2),k] = D_diff[k,:]-s_star
            X[np.arange(1,2*N,2),k] = S[k,:,1]-v_star
            u = -K@X[:,k]    
             
            #error might
            t_x = np.nonzero(u>acel_max)
            if np.all(t_x==0):
                u[u>acel_max] = acel_max
            elif np.all((np.nonzero(u<dcel_max))==0):
                u[u<dcel_max] = dcel_max
                
            for i_AV in range(0,AV_number):
                id_AV = AV_position[0][i_AV]
                flag = (pow(S[k,id_AV,1],2)-pow(S[k,id_AV-1,1],2)) / 2 / (S[k,id_AV-1,0]-S[k,id_AV,0]-sd) > abs(dcel_max)
                if (flag.any()):
                    u[i_AV] = dcel_max
                S[k,id_AV,2] = u[i_AV]
                


    if (k*Tstep>20) and (k*Tstep<22):
        S[k,brakeID,2]=-5

    S[k+1,:,1] = S[k,:,1] + Tstep*S[k,:,2]
    S[k+1,:,0] = S[k,:,0] + Tstep*S[k,:,1]
    
for k in range(NumStep):
    V_avg[k] = np.mean(S[k,:,1])


# In[6] 
#Plot
Lwidth = 1.2;
Wsize = 20;

# In[7] 
# Velocity


#Settling Time
final_velocity = V_avg[NumStep-2]
above_2_percent = final_velocity*1.03
below_2_percent = final_velocity*0.97

settling_time = 0
for k in range(NumStep-2,0,-1):
    for j in range(N):
        if (S[k,j,1] > above_2_percent) or (S[k,j,1] < below_2_percent):
            settling_time = k/100
            break
    if (settling_time != 0):
        break

print("Settling Time within 3% is",settling_time, "s")



#Maximum Spacing in fron of AV
max_space = 0
for k in range(NumStep):
    curr_space = S[k,-2,0]-S[k,-1,0]
    if ( curr_space > max_space):
        max_space = curr_space

print("Maximum Spacing in front of AV is ", round(max_space,2) )

#Average settled velocity
print("Average settled velocity is ", round(np.mean(S[(int((0.9*TotalTime)/Tstep)):,:,1]),2), " m/s")



spacing_or_velocity = 1 # 0 , 1 or 2
#Display data

fig = plt.figure()
x = np.arange(0,NumStep)

# syntax for 3-D projection


#y = np.linspace(0,20,20)

if spacing_or_velocity == 0:
    ax = plt.axes(projection ='3d')
    for i in range(N):
        z = np.ones(NumStep-1)*i

        if i==N-1 and mix==1:
            ax.plot3D(z, x[:-1], S[:-1,i-1,0] - S[:-1,i,0], 'red', linewidth=0.5)
            continue

        if i==0:
            ax.plot3D(z, x[:-1], S[:-1,i-1,0] - S[:-1,i,0] + Circumference, 'blue', linewidth=0.5)
            continue

        ax.plot3D(z, x[:-1], S[:-1,i-1,0] - S[:-1,i,0], 'blue', linewidth=0.5)


    ax.set_yticks(np.linspace(0,NumStep,5))
    ax.set_yticklabels(np.linspace(0,TotalTime,5))
    ax.set_xlabel("Vehicle ID")
    ax.set_ylabel("Time")
    ax.set_zlabel("Spacing from vehicle ahead")
    title = "N=" + str(N) + ", mix=" + str(mix)
    ax.set_title(title)
    plt.show()

if spacing_or_velocity == 1:
    ax = plt.axes(projection ='3d')
    for i in range(N):
        z = np.ones(NumStep-1)*i

        if i==N-1 and mix==1:
            ax.plot3D(z, x[:-1], S[:-1,i,1], 'red', linewidth=0.5)
            continue

        ax.plot3D(z, x[:-1], S[:-1,i,1], 'blue', linewidth=0.5)


    ax.set_yticks(np.linspace(0,NumStep,5))
    ax.set_yticklabels(np.linspace(0,TotalTime,5))
    ax.set_xlabel("Vehicle ID")
    ax.set_ylabel("Time")
    ax.set_zlabel("Vehicle Velocity")
    title = "N=" + str(N) + ", mix=" + str(mix) 
    ax.set_title(title)
    plt.show()



if spacing_or_velocity == 2:

    for i in range(N):
        y = S[:,i,0]#circumference
        y[y>399]=np.nan


        #y = M.array(y)
        #masked_y = M.masked_where(y>399,y)
        z = S[:,i,1]
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)


        lc = LineCollection(segments, array=z, cmap=LinearSegmentedColormap.from_list('rg',["black","r", "orange", "y", "limegreen"], N=256), norm=plt.Normalize(0,25), linewidth=0.5)#, alpha=alpha)

        ax = plt.gca()
        ax.add_collection(lc)

        #colorline(x, S[:,i,0]%circumference, S[:,i,1])
        #plt.scatter(x=x, y=S[:,i,0]%circumference, c=S[:,i,1], s=0.00001, cmap=LinearSegmentedColormap.from_list('rg',["b","r","y","g"], N=256), norm=plt.Normalize(0,15))
        #plt.plot(x, S[:,i,0]%circumference, c=S[:,i,1])

    plt.xlim(20000, 60000)
    plt.ylim(0,400)
    plt.show()


# Animation
from matplotlib import pyplot as plt

x = []
y = []
AV_x = []
AV_y = []
R = Circumference / 2 / math.pi
for id in range(20):
    temp_x = R * math.cos(S[0, id, 0] / Circumference * 2 * math.pi)
    temp_y = R * math.sin(S[0, id, 0] / Circumference * 2 * math.pi)
    if ID[id] == 0:
        x.append(temp_x)
        y.append(temp_y)
    else:
        AV_x.append(temp_x)
        AV_y.append(temp_y)

    # Mention x and y limits to define their range
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)

    if ID[id] == 0:
        plt.scatter(x, y, color='green')
        plt.pause(0.01)
    else:
        plt.scatter(AV_x, AV_y, color='blue')
        plt.pause(0.01)

plt.show()



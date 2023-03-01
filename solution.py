import numpy as np
import random
import matplotlib.pyplot as plt
import numba
import time

def OMP(A, y, s, noise, n_norm):

    N = A.shape[1]
    M = A.shape[0]
    k = []
    a = []
    r = y

    while (max(np.abs(A.T @ r)) > 0.1):
        
        x_omp = np.zeros((N, 1))
        #Find the index of the maximum value in the vector A.T @ b
        j = np.argmax(np.abs(A.T @ r))
        #Add the index to the list k
        k.append(j)
        if type(a) == list:
            a.append(A[:, j].reshape(M, 1))
            a = np.array(a)
            a = a.reshape(M, 1)
        else:
            a = np.concatenate((a, A[:, j].reshape(M, 1)), axis=1)
        a_temp = (a.T) @ a
        a_temp = np.linalg.inv(a_temp)
        alpha = (a_temp @ a.T) @ y
        b = a @ alpha
        #Update r
        r = y - b
        for i in range(len(k)):
            x_omp[k[i]] = alpha[i]
        error = np.linalg.norm(y - (A @ x_omp))
        if noise == True:
            if error < n_norm:
                break
            if s == k:
                break
        if error < 1e-3:
            break

    
    x_omp = np.zeros((N, 1))
    for i in range(len(k)):
        x_omp[k[i]] = alpha[i]
    
    return x_omp, len(k)

N = 100
s_max = 19
A_normal_upper = 1


esr_plot = []
avg_error_plot = []

A = []
"""
for M in range(N):
    if M == 0:
        continue
    
    avg_error_temp = []
    esr_temp = []

    for s in range(s_max):
        if s == 0:
            continue
        
        avg_error = 0
        esr = 0
        print("M = ", M, "s = ", s)
        start = time.time()
        for iter in range(2000):

            A = np.random.normal(0, A_normal_upper, (M, N))
            A = A / np.linalg.norm(A, axis=0)
            x = np.zeros((N, 1))
            
            j = random.sample(range(N), s)                
            x[j] = random.randint(-10, 10)
            while sum(x[j] == 0):
                x[j] = random.randint(-10, 10)
            
            y = A @ x
            x_omp, s_cap = OMP(A, y, 0, False, -1)             #s passed as zero;
            x_diff = x - x_omp                                 #As s is not known in noiseless case
            error = np.linalg.norm(x_diff)
            error = error / np.linalg.norm(x)
            avg_error += error
            if s_cap == s:
                esr += 1
        
        avg_error = avg_error / 2000
        esr = esr / 2000
        end = time.time()
        print("Time taken = ", end - start)
        
        avg_error_temp.append(avg_error)
        esr_temp.append(esr)
    
    avg_error_plot.append(avg_error_temp)
    esr_plot.append(esr_temp)

avg_error_plot = np.array(avg_error_plot)
esr_plot = np.array(esr_plot)
"""
avg_error = 0
M = 50
s = 10
esr = 0
print("M = ", M, "s = ", s)
start = time.time()
for iter in range(2000):

    A = np.random.normal(0, A_normal_upper, (M, N))
    A = A / np.linalg.norm(A, axis=0)
    x = np.zeros((N, 1))
    
    j = random.sample(range(N), s)                
    x[j] = random.randint(-10, 10)
    while sum(x[j] == 0):
        x[j] = random.randint(-10, 10)
    
    y = A @ x
    x_omp, s_cap = OMP(A, y, 0, False, -1)             #s passed as zero;
    x_diff = x - x_omp                                 #As s is not known in noiseless case
    error = np.linalg.norm(x_diff)
    error = error / np.linalg.norm(x)
    avg_error += error
    if s_cap == s:
        esr += 1

avg_error = avg_error / 2000
esr = esr / 2000
end = time.time()
print(avg_error)
print("Time taken = ", end - start)
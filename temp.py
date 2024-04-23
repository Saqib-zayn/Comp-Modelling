# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
#%%
#question 1 - part a
#generate random parameters between 0 and 1
def generate_parameters():
  """
  Generates a random set of parameters for the population model.
  """
  return (
      np.random.uniform(0, 1),
      np.random.uniform(0, 1),
      np.random.uniform(0, 1),
      np.random.uniform(0, 1),
      np.random.uniform(0, 1),
      np.random.uniform(0, 1),
      np.random.uniform(0, 1),
      np.random.uniform(0, 1),
      np.random.uniform(0, 1),
      np.random.uniform(0, 1)
  )
parameters = generate_parameters()
#%%
#question 1 - part b
#defining the system of differential equations
def pop_model(t, state, parameters):
    x, y, z = state
    
    a0, ay, az, b0, bx, bz, c0, cx, cy, cz = parameters
    
    eq1 = -x * (a0 - ay * y - az * z)
    eq2 = -y * (b0 + bx * x - bz * z)
    eq3 = z * (c0 - cx * x - cy * y - cz * z)
    
    return [eq1, eq2, eq3]

#set the initial numbers of whales, fish, and plankton
initial_state = [1, 2, 3]

#intervals of time to solve for
t_span = (0, 10) 

#solve the equation
solution = solve_ivp(pop_model, t_span, initial_state, args=(parameters,), dense_output=True)

#time intervals for evaulation
t_eval = np.linspace(0, 10, 100)

# evaluate at desired time intervals
solution_eval = solution.sol(t_eval)

#extracting the solution
whales = solution_eval[0]
fish = solution_eval[1]
plankton = solution_eval[2]

#plotting the results
plt.figure(figsize=(10, 6)) 

#plot whales
plt.plot(t_eval, whales, label='Whales', color='blue')

#plot fish
plt.plot(t_eval, fish, label='Fish', color='green')

#plot plankton
plt.plot(t_eval, plankton, label='Plankton', color='red')

#titles etc
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Population Evolution of Whales, Fish, and Plankton')
plt.legend()

plt.grid(True)
plt.show()

#%%
#question 1 - part c

#check if any population falls below 2% of its initial value
all_safe = min(whales) >= 0.02 and min(fish) >= 0.04 and min(plankton) >= 0.06

#print result of the previous check
if all_safe:
    print("This is a parameter set where no species are endangered")
    
else:
    print("This parameter set led to endangered species. Rerun for a new set.")
    
#%%
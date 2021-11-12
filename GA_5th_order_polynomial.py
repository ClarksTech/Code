#import required libraries

from random import randint, random
from operator import add
import matplotlib.pyplot as plt
import functools
import numpy as np
import math


#Create a member of the population
def individual(length, min, max):
    return[randint(min,max) for x in range(length)]

#create a number of idividuals
def population(count, length, min, max):
    return[ individual(length, min, max) for x in range(count)]

#determine fitness of individual
def fitness(individual, target):
    delta = np.subtract(target, individual)
    delta2 = np.square(delta)
    ave = np.average(delta2)
    dfitness = math.sqrt(ave)
    return dfitness

#find average fitness of population
def grade(pop, target):
    summed = functools.reduce(add, (fitness(x, target) for x in pop))
    return summed/(len(pop)*1.0)

#evolve the population
def evolve(pop, target, retain=0.2, random_select=0.05, mutate=0.01):
    graded = [(fitness(x, target), x) for x in pop]
    graded = [x[1] for x in sorted(graded)]
    retain_length = int(len(graded)*retain)
    parents = graded[:retain_length]
    #randomly add other individuals to promote genetic diversity
    for individual in graded[retain_length:]:
        if random_select > random():
            parents.append(individual)
    #mutate some individuals
    for individual in parents:
        if mutate > random():
            pos_to_mutate = randint(0, len(individual)-1)
            #mutation is non-ideal as restricts range of possible values
            individual[pos_to_mutate] = randint(min(individual), max(individual))
    #crossover parents to create children
    parents_length = len(parents)
    desired_length = len(pop)-parents_length
    children = []
    while len(children) < desired_length:
        male = randint(0, parents_length-1)
        female = randint(0, parents_length-1)
        if male != female:
            male = parents[male]
            female = parents[female]
            half = len(male)//2
            child = male[:half]+female[half:]
            children.append(child)
    parents.extend(children)
    return parents

#polynomial function to generate target points
def polynomial(a1, a2, a3, a4, a5, a6, x):
    y = a1*x^5 + a2*x^4 + a3*x^3 + a4*x^2 + a5*x + a6
    return [x,y]

#polynomial and random point generation
points = []
points_to_match = 1000
for point in range(points_to_match):
    points.append(polynomial(25, 18, 31, -14, 7, -19, randint(-100, 100)))
print(points)


#example usage
target = [25, 18, 31, -14, 7, -19]
p_count = 5000
i_length = 6
i_min = -50
i_max = 50
generations = 100


p = population(p_count, i_length, i_min, i_max)
print(p)
fitness_history = [grade(p, target),]
for i in range(generations):
    p = evolve(p, target)
    fitness_history.append(grade(p, target))
    #stop the algorithm if suitable solution has been found
    suitable_solution = 0.5
    if grade(p, target) < suitable_solution:
        break

for datum in fitness_history:
    print(p)
    print(datum)
    print(target)
    


plt.plot(fitness_history)
plt.show()

#plot polynomial points to find
#points = np.array(points)
#print(points)
#x = points[:,0]
#y = points[:,1]
#coefficients = np.polyfit(x, y, 6)
#poly = np.poly1d(coefficients)
#new_x = np.linspace(x[0], x[-1])
#new_y = poly(new_x)
#plt.plot(x,y, "o", new_x,new_y)
#plt.xlim(-100, 100)
#plt.show()

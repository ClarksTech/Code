#import required libraries

from random import randint, random
from operator import add
import matplotlib.pyplot as plt
import functools


#Create a member of the population
def individual(length, min, max):
    return[randint(min,max) for x in range(length)]

#create a number of idividuals
def population(count, length, min, max):
    return[ individual(length, min, max) for x in range(count)]

#determine fitness of individual
def fitness(individual, target):
    sum = functools.reduce(add, individual, 0)
    return abs(target-sum)

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

#example usage
target = 550
p_count = 100
i_length = 6
i_min = 0
i_max = 100
generations = 100
p = population(p_count, i_length, i_min, i_max)
fitness_history = [grade(p, target),]
for i in range(generations):
    p = evolve(p, target)
    fitness_history.append(grade(p, target))

for datum in fitness_history:
    print(datum)

plt.plot(fitness_history)
plt.show()

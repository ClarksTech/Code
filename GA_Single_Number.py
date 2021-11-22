#import required libraries

from random import randint, random
from operator import add
import matplotlib.pyplot as plt
import functools
import numpy as np

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
def evolve(pop, target, retain, random_select, mutate):
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

#re-usable function to sweep parameters to see their effect on how quickly a solution is found
def sweep_parameter(p_count, retain, random_select, mutate):

    runs_fitness_history = []
    iterations_needed_history = []
    sum_of_runs = 0
    performance = 0
    sum_of_iterations = 0

    #loop for the number of runs to be averaged as smooths random starting variance
    for x in range(runs_to_average):
        iterations_needed = generations
        #main usage
        p = population(p_count, i_length, i_min, i_max)
        fitness_history = [grade(p, target),]
        for i in range(generations):
            p = evolve(p, target, retain, random_select, mutate)
            fitness_history.append(grade(p, target))
            #stop the algorithm if suitable solution has been found
            suitable_solution = 0
            for indv in p:
                if fitness(indv,target) == suitable_solution:
                    iterations_needed = i
                    solution_found = 1
                    solution = functools.reduce(add, indv, 0)
                else:
                    solution_found = 0

            if solution_found == 1:
                performance = performance + 1
                
                break  
        
        #was solution found
        if solution_found == 1:
            print("Run ", x, " solution found in ",iterations_needed," runs: ", solution)
        if solution_found == 0:
            print("Run ", x, " solution NOT found in ", generations," generations")

        #get sum of iterations for average, store in array
        sum_of_iterations = sum_of_iterations + iterations_needed
        iterations_needed_history.append(iterations_needed)

        #for datum in fitness_history:
            #print(datum)

        #get how many runs were completed total, store fitness for runs
        sum_of_runs = sum_of_runs + fitness_history[-1]
        runs_fitness_history.append(fitness_history[-1])

        #display fitness history for individual run
        if show_generation_fitness_graph == "Y":
            plt.title("Fitness History for each generation")
            plt.xlabel("Generation")
            plt.ylabel('Fitness')
            plt.plot(fitness_history)
            plt.show()

    #display average iterations needed to get solution
    average_iterations = sum_of_iterations/runs_to_average
    print("Average iterations for ", runs_to_average, " runs was: ", average_iterations)
    average_iteration_history.append(average_iterations)

    #display the average fitness of all the runs
    average_fitness = sum_of_runs/runs_to_average
    print("Average Fitness for ", runs_to_average, " runs was: ", average_fitness)

    #display performance of all runs as percentage 
    final_performance = (performance/runs_to_average)*100
    print("The GA converges on the correct answer ", final_performance, "% of the time")

    #plot graph of each runs final fitness to see variance
    if show_average_fitness_variance_graph == "Y":
        plt.title("Population fitness of solution on each run")
        plt.xlabel("Run")
        plt.ylabel('Population Fitness')
        plt.plot(runs_fitness_history)
        plt.show()

    #plot graph of iterations needed to see variance 
    if show_average_iterations_needed_graph == "Y":
        plt.title("Iterations needed to find solution on each run")
        plt.xlabel("Run")
        plt.ylabel('Iterations needed')
        plt.plot(iterations_needed_history)
        plt.show()
    return

#function to plot affects of sweep on number of itterations to find solution
def plot_swept_param(swept, swept_history):
    x = np.array(swept_history)
    y = np.array(average_iteration_history)
    theta = np.polyfit(x, y, 3)
    print(f'The Parameters of the curve: {theta}')
    y_line = theta[3] + theta[2] * pow(x, 1) + theta[1]  * pow(x, 2) +theta[0] * pow(x,3)

    title = "Affect of " + swept + " on how quickly solution is found"
    plt.title(title)
    plt.xlabel(swept)
    plt.ylabel('Average Iterations to find solution')
    plt.scatter(x,y)
    plt.plot(x, y_line, 'r')
    plt.show()
    return

#default values
    #target = 550
    #p_count = 100
    #i_length = 6
    #i_min = 0
    #i_max = 100
    #generations = 100
    #retain=0.2
    #random_select=0.05
    #mutate=0.01

#initialise global variable and get user input
i_length = 6
i_min = 0
i_max = 100
target = 550
generations = 100
runs_to_average = int(input("Enter number of runs to find average fitness from: "))
show_generation_fitness_graph = input("Do you want Generational Fitness Graphs for every run? Y/N: ")
show_average_fitness_variance_graph = input("Do you want to show fitness variance graphs? Y/N: ")
show_average_iterations_needed_graph = input("Do you want to show iterations variance graphs? Y/N: ")

#run code for optimal solution (least number of iterations to converge on solution)
if input("Least Iterations Solution? Y/N: ") == "Y":

    retain=0.15
    random_select=0.05
    mutate=0.03
    p_count = 600
    average_iteration_history = []

    sweep_parameter(p_count, retain, random_select, mutate)

#run code for population sweep to see effect on how quickly solution is found
if input("Sweep population? Y/N: ") == "Y":

    #set values for GA
    p_count_history = []
    average_iteration_history = []
    retain=0.2
    random_select=0.05
    mutate=0.01

    #sweep population
    for p_count in range(100, 1100, 100):
        p_count_history.append(p_count)

        sweep_parameter(p_count, retain, random_select, mutate)
    
    #plot affect
    plot_swept_param("population", p_count_history)

#run code for mutation sweep to see effect on how quickly solution is found
if input("Sweep Mutation probability? Y/N: ") == "Y":

    #set values for GA
    mutate_history = []
    average_iteration_history = []
    p_count = 600
    retain=0.2
    random_select=0.05

    #sweep mutation probability
    for mutate in np.arange(0.01, 0.11, 0.01):
        mutate_history.append(mutate)

        sweep_parameter(p_count, retain, random_select, mutate)
    
    #plot affect
    plot_swept_param("mutation probability", mutate_history)

#run code for crossover sweep to see effect on how quickly solution is found
if input("Sweep crossover probability? Y/N: ") == "Y":

    #set values for GA
    crossover_history = []
    average_iteration_history = []
    p_count = 600
    random_select= 0.05
    mutate=0.03

    #sweep crossover probability
    for retain in np.arange(0.1, 0.4, 0.025):
        crossover_history.append(retain)

        sweep_parameter(p_count, retain, random_select, mutate)
    
    #plot affect
    plot_swept_param("crossover probability", crossover_history)



    
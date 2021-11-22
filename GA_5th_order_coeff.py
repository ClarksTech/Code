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
def evolve(pop, target, retain, random_select, mutate):
    
    #ranked selection
    if selection_method == 1:
        graded = [(fitness(x, target), x) for x in pop]
        graded = [x[1] for x in sorted(graded)]
        retain_length = int(len(graded)*retain)
        #decide if using elitism or not
        if eletism_status == "Y":
            parents = graded[2:(retain_length)]   # 2 top kept out of mutations and crossover elitism
        else:
            parents = graded[:(retain_length)]      # no eletism
    
    #roulette selection
    if selection_method == 2:
        #population fitness
        population_fitness = sum([fitness(x, target) for x in pop])
        #chromosone probability
        chromosone_prob = [fitness(x, target)/population_fitness for x in pop]
       #normalise
        norm_chromosone_prob = []
        max_prob = max(chromosone_prob)
        min_prob = min(chromosone_prob)
        for x in range(len(chromosone_prob)):
            norm_chromosone_prob.append((chromosone_prob[x]-min_prob)/(max_prob - min_prob))
        #convert prob for minaturization
        norm_chromosone_prob = 1 - np.array(norm_chromosone_prob)
        #make probabilities sum to 1
        final_chromosone_probs = []
        for x in range(len(norm_chromosone_prob)):
            final_chromosone_probs.append(norm_chromosone_prob[x]/sum(norm_chromosone_prob))
        #create parents population
        retain_length = int(len(final_chromosone_probs)*retain)
        graded = []
        for x in range(retain_length):
            temp = np.arange(0,len(pop),1) # 1D size of pop as random choice numpy only takes 1D
            roulette_position_selected = np.random.choice(temp, p=final_chromosone_probs)
            graded.append(pop[roulette_position_selected])
        #decide if using elitism or not
        if eletism_status == "Y":
            parents = graded[2:(retain_length)]   # 2 top kept out of mutations and crossover elitism
        else:
            parents = graded[:(retain_length)]      # no eletism

    #randomly add other individuals to promote genetic diversity
    for individual in graded[retain_length:]:
        if random_select > random():
            parents.append(individual)

    #mutate some individuals
    for individual in parents:
        if mutate > random():
            pos_to_mutate = randint(0, len(individual)-1)
            #mutation is non-ideal as restricts range of possible values
            individual[pos_to_mutate] = randint(-50, 50) #in range

    #crossover parents to create children
    parents_length = len(parents)
    #children produced depends on if elite are kept
    if eletism_status == "Y": 
        desired_length = len(pop)-(parents_length+2)
    else:
        desired_length = len(pop)-(parents_length)
    children = []
    while len(children) < desired_length:
        male = randint(0, parents_length-1)
        female = randint(0, parents_length-1)
        if male != female:
            male = parents[male]
            female = parents[female]
            #1-point crossover at random crossover point
            if crossover_method == 1:
                #half = len(male)//2
                crossover_point = randint(1,(len(male)-1))
                child = male[:crossover_point]+female[crossover_point:]
                children.append(child)
            #2-point crossover at random crossover points
            if crossover_method == 2:
                #half = len(male)//2
                crossover_point_1 = randint(1,(len(male)-1))
                crossover_point_2 = randint(1,(len(male)-1))
                #determine which crossover point comes first and add genes to child accordingly
                if (crossover_point_1 < crossover_point_2) & (crossover_point_1 != crossover_point_2):
                    child = male[:crossover_point_1]+female[crossover_point_1:crossover_point_2]+male[crossover_point_2:]
                    children.append(child)
                if (crossover_point_1 > crossover_point_2) & (crossover_point_1 != crossover_point_2):
                    child = male[:crossover_point_2]+female[crossover_point_2:crossover_point_1]+male[crossover_point_1:]
                    children.append(child)
            #uniform crossover
            if crossover_method == 3:
                #generate mask
                unifrom_crossover_mask = np.random.choice([0,1], size=len(male))
                #make child gene from male or female parent depending on mask bit
                child = []
                for x in range(len(unifrom_crossover_mask)):
                    if unifrom_crossover_mask[x] == 0:
                        child.append(male[x])
                    if unifrom_crossover_mask[x] == 1:
                        child.append(female[x])
                children.append(child)

    #if eleitism used add back in most elite
    if eletism_status == "Y":        
        parents.append(graded[0])#add back in most elite
        parents.append(graded[1])#add back in 2nd most elite
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
            #individual solution fitness function termination
            if termination_function == 1:
                suitable_solution = 0
                for indv in p:
                    if fitness(indv,target) == suitable_solution:
                        iterations_needed = i
                        solution_found = 1
                        solution = indv
                    else:
                        solution_found = 0
            #population average fitness function termination           
            if termination_function == 2:
                suitable_solution = 1
                if grade(p, target) < suitable_solution:
                    iterations_needed = i
                    solution_found = 1
                    performance = performance + 1
                    lowest_indv_fitness = 1000000 #arbitrary high
                    for indv in p:
                        if fitness(indv,target) < lowest_indv_fitness:
                            solution = indv
                    break
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
    average_fitness_history.append(average_fitness)
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
    y = np.array(average_fitness_history)
    theta = np.polyfit(x, y, 3)
    print(f'The Parameters of the curve: {theta}')
    y_line = theta[3] + theta[2] * pow(x, 1) + theta[1]  * pow(x, 2) +theta[0] * pow(x,3)

    title = "Affect of " + swept + " on average fitness"
    plt.title(title)
    plt.xlabel(swept)
    plt.ylabel('Average fitness')
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
target = [25, 18, 31, -14, 7, -19]
i_length = 6
i_min = -50
i_max = 50
generations = 100
runs_to_average = int(input("Enter number of runs to find average fitness from: "))
eletism_status = input("Do you want to use Elitism? Y/N: ")
selection_method = int(input("Select selection function: Ranked (1) or Roulette (2): "))
crossover_method = int(input("Select crossover function: 1-point (1) or 2-point (2) or uniform mask (3): "))
termination_function = int(input("Select termination function: Individual (1) or Population (2): "))
show_generation_fitness_graph = input("Do you want Generational Fitness Graphs for every run? Y/N: ")
show_average_fitness_variance_graph = input("Do you want to show fitness variance graphs? Y/N: ")
show_average_iterations_needed_graph = input("Do you want to show iterations variance graphs? Y/N: ")


#run code for optimal solution (least number of iterations to converge on solution)
if input("Least Iterations Solution? Y/N: ") == "Y":

    retain=0.2
    random_select=0.05
    mutate=0.03
    p_count = 5000
    average_iteration_history = []
    average_fitness_history = []

    sweep_parameter(p_count, retain, random_select, mutate)

#run code for population sweep to see effect on how quickly solution is found
if input("Sweep population? Y/N: ") == "Y":

    #set values for GA
    p_count_history = []
    average_iteration_history = []
    average_fitness_history = []
    retain=0.4
    random_select=0.05
    mutate=0.01

    #sweep population
    for p_count in range(1000, 7000, 1000):
        p_count_history.append(p_count)

        sweep_parameter(p_count, retain, random_select, mutate)
    
    #plot affect
    plot_swept_param("population", p_count_history)

#run code for mutation sweep to see effect on how quickly solution is found
if input("Sweep Mutation probability? Y/N: ") == "Y":

    #set values for GA
    mutate_history = []
    average_iteration_history = []
    average_fitness_history = []
    retain=0.2
    random_select=0.05
    p_count = 5000

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
    average_fitness_history = []
    random_select=0.05
    mutate=0.01
    p_count = 5000

    #sweep crossover probability
    for retain in np.arange(0.1, 0.4, 0.025):
        crossover_history.append(retain)

        sweep_parameter(p_count, retain, random_select, mutate)
    
    #plot affect
    plot_swept_param("crossover probability", crossover_history)



    
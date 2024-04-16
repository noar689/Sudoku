import random
import numpy as np
geneset = [1, 2, 3, 4, 5, 6, 7, 8, 9]
mutation_rate=0.01

def get_parents():
    chromosome = []
    for j in range(9):
        row=[]  
        for i in range (9):
            row.append(random.choice(geneset))
        chromosome.append(row)
    return chromosome

def initial_population(population_size):
    population=[]
    for i in range (population_size):
        population.append(get_parents())
    return population


def fitness(chromosome):
    score=0
    for k in range(9):
        score+=len(set(chromosome[k]))
        culomns=[chromosome[j][k] for j in range(9)]
        score+=len(set(culomns))

    #numper of unique numbers in each square
    for i in range (0,9,3):
        for j in range (0,9,3):  
            elemnts=[]
            for k in range(i,i+3):
                for kk in range(j,j+3):
                    elemnts.append(chromosome[k][kk])

            score+=len(set(elemnts))
            
    return score

#one point crossover
# def crossover(parent1, parent2):
#     crossover_point = random.randint(1, 8)
#     child1 = np.vstack((parent1[:crossover_point], parent2[crossover_point:]))
#     child2 = np.vstack((parent2[:crossover_point], parent1[crossover_point:]))
#     return child1,child2

#uniform crossover
def crossover(parent1, parent2):
    child1 = []
    child2 = []
    for i in range(9):
        row1 = []
        row2 = []
        for j in range(9):
            if random.random() < 0.5:
                row1.append(parent1[i][j])
                row2.append(parent2[i][j])
            else:
                row1.append(parent2[i][j])
                row2.append(parent1[i][j])
        child1.append(row1)
        child2.append(row2)
    return child1, child2

def mutation(individual):
    index = random.randrange(0, len(individual))
    child = individual[:]
    new_gene, alter = random.sample(geneset, 2)
    if new_gene == child[index][index] :
        child[index][index]  = alter
    else:
        child[index][index]  = new_gene
    return child
    # for i in range(9):
    #     for j in range(9):
    #         if random.random() < mutation_rate:
    #             individual[i][j] = random.choice(geneset)
    # return individual

def genetic_algorithm(population_size,mutation_rate, max_generations):
    population = initial_population(population_size)
    for generation in range(max_generations):
        fitness_score=[fitness(indivdual) for indivdual in population]
        max_f=sorted(fitness_score ,reverse=True)[0]
        population = sorted(population, key=fitness, reverse=True)
        best_indivdual=population[0]
        max_f_index=fitness_score.index(max_f)
        best_fitness=fitness_score[max_f_index]

        print(f'generation {generation} , best indivdual {best_indivdual} , with fitness {best_fitness}\n')
        
        
        next_generation = [population[0]]
        while len(next_generation) < population_size:
            parent1 =population[max_f_index]
            parent2 =population[max_f_index+1]

            child1,child2 = crossover(parent1, parent2)
            child1 = mutation(child1)
            child2 = mutation(child2)
            #replacement
            next_generation.append(child1)
            if fitness(child2) > fitness(parent2):
                next_generation.append(child2)
            next_generation.append(parent2)
        population = next_generation
    
    # return max(population,key=fitness)


solution = genetic_algorithm(population_size=10,mutation_rate=0.01, max_generations=30)
print(solution)


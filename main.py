import math
import random

import numpy as np
import matplotlib.pyplot as plt

def split_list(lst):
    if len(lst) == 0:
        return []
    if len(lst) == 3:
        return [lst]
    else:
        return [[lst[0], lst[1]]] + split_list(lst[2:])
        
def binary_search(arr, val):
    st = 0
    dr = len(arr) - 1
    rasp = 0
    while st < dr:
        mid = (st + dr) // 2
        if arr[mid] <= val:
            rasp = mid
            st = mid + 1
        else:
            dr = mid - 1
            
    return rasp

class Logger:
    logFile = None

    @classmethod
    def set(cls, f):
        cls.logFile = f

    @classmethod
    def log(cls, msg):
        print(msg, file=cls.logFile)

class Chromosome:
    fitness = None
    a = None
    b = None
    length = None
    mutation_p = None

    def __init__(self, val=None):
        if val is None:
            self.chromosome = ''.join(np.random.choice(['0', '1'], p=[0.5, 0.5], size=Chromosome.length))
        else:
            self.chromosome = val

    def decode(self):
        return int(self.chromosome, 2) * (Chromosome.b - Chromosome.a) / (2 ** Chromosome.length - 1) + Chromosome.a

    @classmethod
    def set(cls, fit, aa, bb, leng, mutation):
        cls.fitness = fit
        cls.a = aa
        cls.b = bb
        cls.length = leng
        cls.mutation_p = mutation

    def fitness_score(self):
        return Chromosome.fitness(self.decode())

    def __lt__(self, other):
        return self.fitness_score() <= other.fitness_score()

    def chromo_crossover(self, other, i, j):
        crossover_point = random.randint(0, Chromosome.length)
        if Generation.first_generation:
            Logger.log("Recombinare dintre cromozomul {} cu cromosomul {}:".format(i, j))
            Logger.log("{} {} punct {}".format(self.chromosome, other.chromosome, crossover_point))
            Logger.log("Rezultat: {} {}".format(self.chromosome[:crossover_point] + other.chromosome[crossover_point:], other.chromosome[:crossover_point] + self.chromosome[crossover_point:]))
        return Chromosome(self.chromosome[:crossover_point] + other.chromosome[crossover_point:]), Chromosome(other.chromosome[:crossover_point] + self.chromosome[crossover_point:])

    def chromo_mutate(self, index):
        lst = list(self.chromosome)
        ok = False
        for i in range(len(lst)):
            u = np.random.uniform(0, 1)
            if u < Chromosome.mutation_p:
                ok = True
                if lst[i] == '0':
                    lst[i] = '1'
                else:
                    lst[i] = '0'
        if Generation.first_generation and ok:
            Logger.log(index)
        self.chromosome = ''.join(lst)



class Generation:
    first_generation = True
    population_count = None
    cross_p = None

    def __init__(self, pop=None):
        if pop is None:
            self.population = [Chromosome() for _ in range(Generation.population_count)]
        else:
            self.population = pop

    @classmethod
    def set(cls, pop, cross):
        cls.population_count = pop
        cls.cross_p = cross


    def roulette_selection(self, n):
        total_f = sum([x.fitness_score() for x in self.population])
        p = [x.fitness_score() / total_f for x in self.population]
        q = [0]
        cumulatibe_sum = 0
        for i in range(len(p)):
            cumulatibe_sum += p[i]
            q.append(cumulatibe_sum)

        if Generation.first_generation:
            Logger.log("Probabilitati selectie")
            for i in range(len(self.population)):
                Logger.log("Chromosome {}: probabilitate selectie: {}".format(i, p[i]))

        if Generation.first_generation:
            Logger.log("Intervale probabilitati  selectie")
            for x in q:
                Logger.log(x)
        new_pop = []
        for _ in range(n):
            u1 = np.random.uniform(0, 1)
            u2 = np.random.uniform(0, 1)

            rasp1 = binary_search(q, u1)
            rasp2 = binary_search(q, u2)
            
            
            if self.population[rasp1] < self.population[rasp2]:
                new_pop.append(self.population[rasp2])
            else:
                new_pop.append(self.population[rasp1])
            

            #new_pop.append(self.population[rasp])
            if Generation.first_generation:
                pass
                #Logger.log("u={}, selected chromozome: {}".format(u, rasp))
        return new_pop

    def elitist_selection(self, n):
        pop_copy = sorted(self.population)
        if Generation.first_generation:
            Logger.log("Eelementele eliste alese {}:".format(n))
            for x in pop_copy[-n:]:
                Logger.log("{} fitness {}".format(x.chromosome, x.fitness_score()))
        return pop_copy[-n:]

    def selection(self):
        return self.roulette_selection(self.population_count-1)

    def crossover(self):
        new_pop = self.population.copy()
        cross_over_index = []
        if Generation.first_generation:
            Logger.log("Probabilitate de incrucisare {}".format(Generation.cross_p))
        for i in range(Generation.population_count-1):
            u = np.random.uniform(0, 1)
            if u < Generation.cross_p:
                if Generation.first_generation:
                    Logger.log("Chromosome {}: u = {} < {} participa".format(i, u, Generation.cross_p))
                cross_over_index.append(i)
            else:
                if Generation.first_generation:
                    Logger.log("Chromosome {}: u = {}".format(i, u))

        if len(cross_over_index) < 2:
            return new_pop

        to_corss_over = split_list(cross_over_index)
        for indices in to_corss_over:
            if len(indices) == 2:
                new_pop[indices[0]], new_pop[indices[1]] = new_pop[indices[0]].chromo_crossover(new_pop[indices[1]], indices[0], indices[1])
            else:
                new_pop[indices[0]], _ = new_pop[indices[0]].chromo_crossover(new_pop[indices[1]],indices[0], indices[1])
                new_pop[indices[1]], _ = new_pop[indices[1]].chromo_crossover(new_pop[indices[2]], indices[1], indices[2])
                new_pop[indices[2]], _ = new_pop[indices[2]].chromo_crossover(new_pop[indices[0]], indices[2], indices[0])

        return new_pop


    def mutation(self):
        if Generation.first_generation:
            Logger.log("Probabilitatea de mutatie pentru fiecare gena {}".format(Chromosome.mutation_p))
            Logger.log("Au fost modificati cromozomii: ")
        for i in range(len(self.population)):
            self.population[i].chromo_mutate(i)
        return self.population

    def next_generation(self):
        if Generation.first_generation:
            Logger.log('Populatie:')
            for i in range(len(self.population)):
                Logger.log('Chromosome {}: {}, Value: {}, Fitness: {}'.format(i, self.population[i].chromosome, self.population[i].decode(), self.population[i].fitness_score()))

        selection = Generation(self.selection())
        if Generation.first_generation:
            Logger.log('Dupa selectie:')
            for i in range(len(selection.population)):
                Logger.log('Chromosome {}: {}, Value: {}, Fitness: {}'.format(i, selection.population[i].chromosome, selection.population[i].decode(), selection.population[i].fitness_score()))

        crossover = Generation(selection.crossover())
        if Generation.first_generation:
            Logger.log('Dupa recombinare:')
            for i in range(len(crossover.population)):
                Logger.log('Chromosome {}: {}, Value: {}, Fitness: {}'.format(i, crossover.population[i].chromosome, crossover.population[i].decode(), crossover.population[i].fitness_score()))

        mutation = Generation(crossover.mutation())


        if Generation.first_generation:
            Logger.log('Dupa mutatie:')
            for i in range(len(crossover.population)):
                Logger.log('Chromosome {}: {}, Value: {}, Fitness: {}'.format(i, mutation.population[i].chromosome, mutation.population[i].decode(), mutation.population[i].fitness_score()))
        mutation.population = mutation.population + self.elitist_selection(1)
        self.population = mutation.population.copy()


pop_count = int(input())
a, b = list(map(float, input().split()))
x2, x1, x0 = list(map(float, input().split()))
p = int(input())
crossover_prob = float(input())
mutation_prob = float(input())
nr_iter = int(input())

chromozome_length = math.ceil(math.log2((b - a) * 10 ** p))

Chromosome.set(lambda x: -x ** 3 + 3 * x ** 2 + 6 * x + 3, a, b, chromozome_length, mutation_prob)
Generation.set(pop_count, crossover_prob)

gen = Generation()

output = open("output.txt", "w")
Logger.set(output)

fig, axs = plt.subplots(3)

maxis_x = []
maxis_y = []

avg_x = []
avg_y = []

for i in range(nr_iter):
    gen.next_generation()
    Generation.first_generation = False
    fitness_list = [x.fitness_score() for x in gen.population]
    vals = [x.decode() for x in gen.population]
    

    x = np.linspace(a, b, 1000)
    f = lambda x: -x ** 3 + 3 * x ** 2 + 6 * x + 3
    y = [f(a) for a in x]
    axs[0].cla()
    axs[1].cla()
    axs[2].cla()
    axs[0].plot(x, y)
    axs[0].scatter(vals, fitness_list, color='black')

    maxim_ind = np.argmax(fitness_list)
    axs[0].scatter(vals[maxim_ind], fitness_list[maxim_ind], color='red')
    maxis_x.append(i)
    maxis_y.append(max(fitness_list))
    avg_x.append(i)
    avg_y.append(sum(fitness_list) / len(fitness_list))

    axs[1].plot(maxis_x, maxis_y)
    axs[2].plot(avg_x, avg_y)

    plt.pause(0.05)
    Logger.log("Valoarea maxima: {}, Valoarea medie: {}".format(max(fitness_list), sum(fitness_list) / len(fitness_list)))

fitness_list = [x.fitness_score() for x in gen.population]
print("Maximul este {} in punctul {}".format(max(fitness_list), gen.population[np.argmax(fitness_list)].decode()))
plt.show()

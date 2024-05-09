'''
Genetic Algorithm for evolving Izhikevich networks
Justin Slattery, 2024
'''
import numpy as np
import random
from izhikevich_nn import IzhikevichNetwork

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, elitism_rate, num_generations, num_excitatory, network_size, fitness_function):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.num_generations = num_generations
        self.num_excitatory = num_excitatory
        self.network_size = network_size
        self.fitness_function = fitness_function
        self.population = [self.create_network() for _ in range(population_size)]
        self.fitness_scores = np.zeros(population_size)

    def create_network(self):
        network = IzhikevichNetwork(self.network_size)
        network.pcnnParams(self.num_excitatory)
        return network

    def evaluate_population(self):
        self.fitness_scores = np.array([self.fitness_function(network) for network in self.population])

    def select_best_networks(self):
        elite_count = int(self.population_size * self.elitism_rate)
        best_indices = np.argsort(-self.fitness_scores)[:elite_count]
        return [self.population[i] for i in best_indices]

    def mutate_network(self, network):
        for i in range(network.Size):
            if random.random() < self.mutation_rate:
                network.a[i] += np.random.normal(0, 0.01)
                network.b[i] += np.random.normal(0, 0.01)
                network.c[i] += np.random.normal(0, 1)
                network.d[i] += np.random.normal(0, 1)

            if random.random() < self.mutation_rate:
                network.Weights[i] += np.random.normal(0, 0.1, size=network.Size)

        network.Weights = np.clip(network.Weights, -1, 1)
        return network

    def generate_new_population(self, best_networks):
        new_population = best_networks[:]
        while len(new_population) < self.population_size:
            parent = random.choice(best_networks)
            child = self.create_network()
            child.a = np.copy(parent.a)
            child.b = np.copy(parent.b)
            child.c = np.copy(parent.c)
            child.d = np.copy(parent.d)
            child.Weights = np.copy(parent.Weights)
            child = self.mutate_network(child)
            new_population.append(child)
        return new_population

    def evolve(self, callback=None):
        for generation in range(self.num_generations):
            #print(f"Generation {generation}")
            self.evaluate_population()
            best_networks = self.select_best_networks()
            self.population = self.generate_new_population(best_networks)

            if callback:
                callback()

        # Evaluate the final population and return the best network
        self.evaluate_population()
        best_network = self.select_best_networks()[0]
        return best_network

# In the main simulation file:
# Define the fitness function and create the GA instance
# fitness_function = lambda network: # Your fitness calculation logic
# ga = GeneticAlgorithm(population_size=10, mutation_rate=0.1, elitism_rate=0.2, num_generations=50,
#                       num_excitatory=800, network_size=1000, fitness_function=fitness_function)
# best_network = ga.evolve()



''' 
Simulation file for evolving a two-node IZ network as a coupled oscillator
'''

from ga import GeneticAlgorithm
import numpy as np
import matplotlib.pyplot as plt
from alive_progress import alive_bar

# Define the fitness function and create the GA instance
# Create a placeholder fitness function for now

# Fitness function to evaluate all networks in the population
def custom_fitness_function(network):
    dt = 0.1
    steps = 1000
    inputs = np.random.normal(10, 2, size=(steps, network.Size))
    spikes = np.zeros((steps, network.Size))

    for t in range(steps):
        network.step(dt, inputs[t])
        spikes[t] = network.Firing

    total_spikes = np.sum(spikes)
    return total_spikes

# Example usage
population_size = 100
mutation_rate = 0.05
elitism_rate = 0.2
num_generations = 10
num_excitatory = 70
network_size = 100

# Initialize the GeneticAlgorithm with the fitness function
ga = GeneticAlgorithm(
    population_size=population_size,
    mutation_rate=mutation_rate,
    elitism_rate=elitism_rate,
    num_generations=num_generations,
    num_excitatory=num_excitatory,
    network_size=network_size,
    fitness_function=custom_fitness_function
)

# Evolve the best network
with alive_bar(num_generations, title="Evolving Networks") as bar:
    best_network = ga.evolve(callback=bar)

# Plot the best network's activity
plt.figure()
plt.plot(best_network.Firing)
plt.title("Best Network Activity")
plt.xlabel("Time Steps")
plt.ylabel("Firing Rate")

# plot the change in fitness scores over generations
plt.figure()
plt.plot(ga.fitness_scores)
plt.title("Fitness Scores Over Generations")
plt.xlabel("Generation")
plt.ylabel("Fitness Score")

plt.show()

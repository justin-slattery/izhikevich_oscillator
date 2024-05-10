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
    # Initialize external inputs
    inputs = np.random.normal(10, 2, size=(steps, network.Size))
    # Initialize input array (to store custom input based on neuron outputs)
    #inputs = np.zeros(network.Size)
    spikes = np.zeros((steps, network.Size))
    voltage_history = np.zeros((steps, network.Size))

    for t in range(steps):
        # Two different coupling scenarios
        
        # In this case, just their spiking
        # Set input based on spikes of both neurons
        # if t > 0 and spikes[t-1, 0] == 1:
        #     inputs[t, 1] = 10  # Set input to second neuron if first neuron spiked in previous step
        # if t > 0 and spikes[t-1, 1] == 1:
        #     inputs[t, 0] = 10  # Set input to first neuron if second neuron spiked in previous step

        # In this case, their spiking and their outputs (voltages)
        if t > 0 and spikes[t-1, 0] == 1:
            inputs[t, 1] += voltage_history[t-1, 0]  # Set input to neuron 1 equal to output of neuron 0 if neuron 0 spiked
        if t > 0 and spikes[t-1, 1] == 1:
            inputs[t, 0] += voltage_history[t-1, 1]  # Set input to neuron 0 equal to output of neuron 1 if neuron 1 spiked

        # Step the network with adjusted inputs
        network.step(dt, inputs[t])

        # Record spikes and voltages
        spikes[t] = network.Firing
        voltage_history[t] = network.Voltages
    
    # Check if there are no spikes in either neuron
    if np.sum(spikes[:, 0]) == 0 or np.sum(spikes[:, 1]) == 0:
        return 0, voltage_history  # Return default fitness value

    # Compute synchronization metric (cross-correlation coefficient)
    sync_metric = np.corrcoef(spikes[:, 0], spikes[:, 1])[0, 1]

    # Check for zero standard deviations to avoid division by zero
    if np.std(spikes[:, 0]) == 0 or np.std(spikes[:, 1]) == 0:
        return 0, voltage_history  # Return default fitness value

    # Calculate fitness as the inverse of the absolute difference from perfect synchronization (1 - |sync_metric|)
    fitness = 1 - np.abs(sync_metric)

    return fitness, voltage_history

    # total_spikes = np.sum(spikes)
    # return total_spikes, voltage_history

# Example usage
population_size = 100
mutation_rate = 0.05
elitism_rate = 0.2
num_generations = 5
network_size = 2
num_excitatory = network_size

# Initialize the GeneticAlgorithm with the fitness function
ga = GeneticAlgorithm(
    population_size=population_size,
    mutation_rate=mutation_rate,
    elitism_rate=elitism_rate,
    num_generations=num_generations,
    network_size=network_size,
    num_excitatory=num_excitatory,
    fitness_function=custom_fitness_function
)

# Evolve the best network
with alive_bar(num_generations, title="Evolving Networks") as bar:
    best_network = ga.evolve(callback=bar)

# Print the best network's fitness score
# print(f"Best network fitness score: {max(ga.best_fitness_scores)}")

# Plot the voltages of the neurons in the best network over time using the best voltage history
plt.figure()
plt.plot(ga.best_V_history)
plt.title("Voltage of Neurons Over Time")
plt.xlabel("Time Step")
plt.ylabel("Voltage")
plt.legend([f"Neuron {i+1}" for i in range(network_size)])

# plot the change in fitness scores over generations
plt.figure()
plt.plot(ga.best_fitness_scores)
plt.title("Best Fitness Scores Over Generations")
plt.xlabel("Generation")
plt.ylabel("Fitness Score")

plt.show()

from search import *
from random import randint
from assignment2aux import *

def read_tiles_from_file(filename):
    lines = [line.rstrip('\n') for line in open(filename, 'r').readlines()]
    character_to_tile = {' ': (), 'i': (0,), 'L': (0, 1), 'I': (0, 2), 'T': (0, 1, 2)}
    return tuple(tuple(character_to_tile[character] for character in line) for line in lines)

class KNetWalk(Problem):
    def __init__(self, tiles):
        if type(tiles) is str:
            self.tiles = read_tiles_from_file(tiles)
        else:
            self.tiles = tiles
        height = len(self.tiles)
        width = len(self.tiles[0])
        self.max_fitness = sum(sum(len(tile) for tile in row) for row in self.tiles)
        super().__init__(self.generate_random_state())

    def generate_random_state(self):
        height = len(self.tiles)
        width = len(self.tiles[0])
        return [randint(0, 3) for _ in range(height) for _ in range(width)]

    def actions(self, state):
        height = len(self.tiles)
        width = len(self.tiles[0])
        return [(i, j, k) for i in range(height) for j in range(width) for k in [0, 1, 2, 3] if state[i * width + j] != k]

    def result(self, state, action):
        pos = action[0] * len(self.tiles[0]) + action[1]
        return state[:pos] + [action[2]] + state[pos + 1:]

    def goal_test(self, state):
        return self.value(state) == self.max_fitness

    # Returns an integer fitness value of a given state.
    def value(self, state):
        
        i = 0
        height = len(self.tiles)
        width = len(self.tiles[0])
        
        #Construct a nested tuple representation of the state by combining the original tile layout and any rotations applied
        map = [[0 for _ in range(width)] for _ in range(height)]
        for row in self.tiles:
            j = 0
            for tile in row:
                map[i][j] = tuple((con + state[i * width + j]) % 4 for con in tile)
                j += 1
            i += 1
            
        i = 0
        fitness_value = 0
        #Check for any connections between tiles
        for row in map:
            j = 0
            for tile in row:
                # If there is a tile above
                if (i - 1) >= 0:
                    if 1 in map[i][j] and 3 in map[i - 1][j]:
                        fitness_value += 1
                # If there is a tile below
                if (i + 1) < height:
                    if 3 in map[i][j] and 1 in map[i + 1][j]:
                        fitness_value += 1
                # If there is a tile to the left
                if (j - 1) >= 0:
                    if 2 in map[i][j] and 0 in map[i][j - 1]:
                        fitness_value += 1
                # If there is a tile to the right
                if (j + 1) < width:
                    if 0 in map[i][j] and 2 in map[i][j + 1]:
                        fitness_value += 1
                j += 1
            i += 1
                
        return fitness_value

# Configuring an exponential schedule for simulated annealing.
sa_schedule = exp_schedule(k=10, lam=0.1, limit=1000)

# Configuring parameters for the genetic algorithm.
pop_size = 20
num_gen = 1000
mutation_prob = 0.1


def local_beam_search(problem, population):

    #Nesting the list of states passed in as a parameter in Nodes to allow use of expand function (and sorting them from highest to lowest fitness)
    population_size = len(population)
    current_pop = [Node(state) for state in population]
    current_pop = sorted(current_pop, key=lambda node: problem.value(node.state), reverse = True)
    
    while True:
        next_pop = [] 
        #Generate all child states
        for node in current_pop:
            next_pop.extend(node.expand(problem))
        #If there are no child states (i.e at a terminal node, return fittest state in current population)
        if not next_pop:
            break
        #Sort the child states from highest fitness to lowest fitness
        next_pop = sorted(next_pop, key=lambda node: problem.value(node.state), reverse=True)[:population_size]
        #If the fittest child state is not fitter than the fittest state in the current population, return the fittest state from the current population
        if problem.value(next_pop[0].state) <= problem.value(current_pop[0].state):
            break
        current_pop = next_pop
    return current_pop[0].state


def stochastic_beam_search(problem, population, limit=1000):
    
    #Nesting the list of states passed in as a parameter in Nodes to allow use of expand function (and sorting them from highest to lowest fitness)
    population_size = len(population)
    current_pop = [Node(state) for state in population]
    current_pop = sorted(current_pop, key=lambda node: problem.value(node.state), reverse = True)
    
    for i in range(limit):
        next_pop = [] 
        #Generate all child states
        for node in current_pop:
            next_pop.extend(node.expand(problem))
        #If there are no child states (i.e at a terminal node, return fittest state in current population)
        if not next_pop:
            break
        #Get the fitness values of each child state, storing it in a float array
        fitness_values = np.array([problem.value(node.state) for node in next_pop], dtype=float)
        #Get the sum of fitness values
        fitness_sum = np.sum(fitness_values)
        #Divide each fitness value by the sum of fitness values to get the weighted probabilities
        probabilities = fitness_values / fitness_sum
        
        #Choose population_size states from the child population using weighted random sampling 
        next_pop = np.random.choice(next_pop, population_size, False, probabilities)
        #Sort from highest to lowest fitness
        next_pop = sorted(next_pop, key=lambda node: problem.value(node.state), reverse=True)
        
        #Return a goal state if found (i.e. the highest fitness child = max fitness)
        if network.goal_test(next_pop[0].state):
            return next_pop[0].state
        
        current_pop = next_pop
    return current_pop[0].state

if __name__ == '__main__':

    network = KNetWalk('assignment2config.txt')
    visualise(network.tiles, network.initial)

    # Task 1 test code

    run = 0
    method = 'hill climbing'
    while True:
        network = KNetWalk('assignment2config.txt')
        state = hill_climbing(network)
        if network.goal_test(state):
            break
        else:
            print(f'{method} run {run}: no solution found')
            print(f'best state fitness {network.value(state)} out of {network.max_fitness}')
            visualise(network.tiles, state)
        run += 1
    print(f'{method} run {run}: solution found')
    visualise(network.tiles, state)


    # Task 2 test code
    
    run = 0
    method = 'simulated annealing'
    while True:
        network = KNetWalk('assignment2config.txt')
        state = simulated_annealing(network, schedule=sa_schedule)
        if network.goal_test(state):
            break
        else:
            print(f'{method} run {run}: no solution found')
            print(f'best state fitness {network.value(state)} out of {network.max_fitness}')
            visualise(network.tiles, state)
        run += 1
    print(f'{method} run {run}: solution found')
    visualise(network.tiles, state)
    

    # Task 3 test code
    
    run = 0
    method = 'genetic algorithm'
    while True:
        network = KNetWalk('assignment2config.txt')
        height = len(network.tiles)
        width = len(network.tiles[0])
        state = genetic_algorithm([network.generate_random_state() for _ in range(pop_size)], network.value, [0, 1, 2, 3], network.max_fitness, num_gen, mutation_prob)
        if network.goal_test(state):
            break
        else:
            print(f'{method} run {run}: no solution found')
            print(f'best state fitness {network.value(state)} out of {network.max_fitness}')
            visualise(network.tiles, state)
        run += 1
    print(f'{method} run {run}: solution found')
    visualise(network.tiles, state)
    

    # Task 4 test code
    
    run = 0
    method = 'local beam search'
    while True:
        network = KNetWalk('assignment2config.txt')
        height = len(network.tiles)
        width = len(network.tiles[0])
        state = local_beam_search(network, [network.generate_random_state() for _ in range(100)])
        if network.goal_test(state):
            break
        else:
            print(f'{method} run {run}: no solution found')
            print(f'best state fitness {network.value(state)} out of {network.max_fitness}')
            visualise(network.tiles, state)
        run += 1
    print(f'{method} run {run}: solution found')
    visualise(network.tiles, state)
    
    
    # Task 5 test code
    
    run = 0
    method = 'stochastic beam search'
    while True:
        network = KNetWalk('assignment2config.txt')
        height = len(network.tiles)
        width = len(network.tiles[0])
        state = stochastic_beam_search(network, [network.generate_random_state() for _ in range(100)])
        if network.goal_test(state):
            break
        else:
            print(f'{method} run {run}: no solution found')
            print(f'best state fitness {network.value(state)} out of {network.max_fitness}')
            visualise(network.tiles, state)
        run += 1
    print(f'{method} run {run}: solution found')
    visualise(network.tiles, state)
import networkx as nx # for node graph
import matplotlib.pyplot as plt


def create_city_graph(city_list):

    city_list = city_list
    
    city_graph = [(140, 93), (93, 100), (93, 120), (100, 80), (80, 140), 
                  (120, 75), (75, 85), (85, 140), (140, 150), (150, 180), 
                  (80, 130), (100, 95), (150, 145), (75, 60), (130, 95), 
                  (85, 65), (100, 110), (110, 93), (110, 20), (130, 150), 
                  (120, 125), (125, 60), (60, 40), (140, 145), (145, 85)]

    G = nx.Graph()
    G.add_edges_from(city_graph)
    nx.draw(G, with_labels=True, node_color=['red'], node_size=[1000, 600, 800])
    
    plt.show()
    
    return city_graph
    
def plot_distance_fitness_progress(algo, population, population_size, elite_size, mutation_rate):
    
    population = algo.generate_population(population, population_size)
    
    progress = []
    progress.append(1 / algo.get_rank_routes(population)[0][1])
    
    for i in range(0, algo.num_generations):
        population = algo.create_next_generation(population, elite_size, mutation_rate)
        progress.append(1 / algo.get_rank_routes(population)[0][1])
    
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    
    plt.savefig('./images/plot_distance_fitness_progress.png')
    
    plt.show()
    
import numpy as np
import random
import heapq
#------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches 
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
#------------------------------------------------
""" 
   
   Problema de Simulação de Rota Otimizada para Captura de Pokémon usando Algoritmo Genético e A*.
   
                                                                             """
                                         
                        # ---"""" Algoritmo A* para Pathfinding ---


def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    #GEOMETRIA DISCRETA

def reconstruct_path(came_from, current):
    path = [current]
    while came_from.get(current) is not None:
        current = came_from[current]
        path.append(current)
    return path[::-1]

def calcular_custo_a_estrela(grid, start_point, end_point, TERRAIN_COSTS):
    rows, cols = grid.shape

    priority_queue = [(0, 0, start_point)]
    g_score = { (r, c): float('inf') for r in range(rows) for c in range(cols) }
    g_score[start_point] = 0
    came_from = {}
    came_from[start_point] = None

    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)] 

    while priority_queue:
        f_score, current_g, current_point = heapq.heappop(priority_queue)

        if current_point == end_point:
            path = reconstruct_path(came_from, end_point)
            return current_g, path

        r, c = current_point
        for dr, dc in neighbors:
            next_r, next_c = r + dr, c + dc
            next_point = (next_r, next_c)

            if 0 <= next_r < rows and 0 <= next_c < cols:
                terrain_type = grid[next_r, next_c]
                move_cost = TERRAIN_COSTS.get(terrain_type, 1) 
                new_g_score = current_g + move_cost 

                if new_g_score < g_score[next_point]:
                    g_score[next_point] = new_g_score
                    came_from[next_point] = current_point 
                    h_score = manhattan_distance(next_point, end_point)
                    f_score = new_g_score + h_score
                    heapq.heappush(priority_queue, (f_score, new_g_score, next_point))
    return float('inf'), [] 

#----------------------------------------------------------------------------------------------------------



# --- Algoritmo Genético (TSP com Recompensas) ---
class GeneticAlgorithmTSP:
    def __init__(self, outbreak_points, costs_matrix, rewards_list, population_size=150, generations=300):
        self.points = outbreak_points
        self.num_points = len(outbreak_points)
        self.costs_matrix = costs_matrix
        self.rewards_list = rewards_list
        self.POP_SIZE = population_size
        self.GENERATIONS = generations
        self.population = self._initialize_population()
        self.best_route = None
        self.best_fitness = -float('inf')
        self.history = []

    def _initialize_population(self):
        population = []
        base_route = list(range(self.num_points))
        for _ in range(self.POP_SIZE):
            route = base_route[:]
            random.shuffle(route)
            population.append(route)
        return population

    def _calculate_fitness(self, route):
        total_cost = 0
        total_reward = 0
        for i in range(self.num_points):
            start_index = route[i]
            end_index = route[(i + 1) % self.num_points]
            total_cost += self.costs_matrix[start_index, end_index]
        
        for index in route:
            total_reward += self.rewards_list[index]
        
        if total_cost == 0:
            return 0 
        return total_reward / total_cost

    def _selection(self, population_fitness):
        pool_indices = random.sample(range(len(population_fitness)), min(3, len(population_fitness))) 
        best_index_in_pool = pool_indices[0]
        for idx in pool_indices:
            if population_fitness[idx] > population_fitness[best_index_in_pool]:
                best_index_in_pool = idx
        return self.population[best_index_in_pool]

    def _crossover_order(self, parent1, parent2):
        size = len(parent1)
        a, b = random.sample(range(size), 2)
        start, end = min(a, b), max(a, b)

        child1 = [None] * size
        child2 = [None] * size

        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]

        def fill_gaps(child, parent_to_fill_from):
            fill_values = [item for item in parent_to_fill_from if item not in child]
            fill_idx = 0
            for i in range(size):
                if child[i] is None:
                    child[i] = fill_values[fill_idx]
                    fill_idx += 1
            return child

        return fill_gaps(child1, parent2), fill_gaps(child2, parent1)

    def _mutation(self, route, mutation_rate=0.05):
        if random.random() < mutation_rate:
            idx1, idx2 = random.sample(range(self.num_points), 2)
            route[idx1], route[idx2] = route[idx2], route[idx1]
        return route

    def run(self):
        for generation in range(self.GENERATIONS):
            population_fitness = [self._calculate_fitness(route) for route in self.population]
            
            max_fitness_gen = -float('inf')
            best_route_gen = None
            for i, fitness in enumerate(population_fitness):
                if fitness > max_fitness_gen:
                    max_fitness_gen = fitness
                    best_route_gen = self.population[i]

            if max_fitness_gen > self.best_fitness:
                self.best_fitness = max_fitness_gen
                self.best_route = best_route_gen
            self.history.append(max_fitness_gen)

            new_population = []
            while len(new_population) < self.POP_SIZE:
                parent1 = self._selection(population_fitness)
                parent2 = self._selection(population_fitness)
                
                child1, child2 = self._crossover_order(parent1, parent2)
                
                new_population.append(self._mutation(child1))
                if len(new_population) < self.POP_SIZE:
                    new_population.append(self._mutation(child2))
            self.population = new_population
        return self.best_route, self.best_fitness

#----------------------------------------------------------------------------------------------------------

# --- Configurações do Mapa e Pokémon ---
GRID_SIZE = 50
# 0: Planície (Verde), 1: Água (Azul), 2: Montanha (Cinza), 3: Floresta (Verde Escuro), 4: Lava (Laranja)
TERRAIN_GRID = np.random.choice([0, 1, 2, 3, 4], size=(GRID_SIZE, GRID_SIZE), p=[0.5, 0.2, 0.15, 0.1, 0.05])

TERRAIN_COSTS = {
    0: 1,  # Grama/Planície: Custo 1
    1: 3,  # Água Rasa: Custo 3
    2: 7,  # Montanha/Rocha: Custo 7
    3: 2,  # Floresta/Mato Alto: Custo 2 
    4: 10, # Lava/Terreno Hostil: Custo 10
}

# Definições de cores e nomes para a legenda
TERRAIN_LEGENDS = {
    0: {'color': '#8BC34A', 'name': 'Planície/Grama (Custo 1)'},
    1: {'color': '#2196F3', 'name': 'Água Rasa (Custo 3)'},
    2: {'color': '#616161', 'name': 'Montanha/Rocha (Custo 7)'},
    3: {'color': '#4CAF50', 'name': 'Floresta (Custo 2)'},
    4: {'color': '#FF5722', 'name': 'Lava/Hostil (Custo 10)'}
}

pokemon_colors = [TERRAIN_LEGENDS[i]['color'] for i in range(5)]
cmap_pokemon = ListedColormap(pokemon_colors)

OUTBREAK_POINTS = [
    (5, 5), (10, 45), (25, 15), (40, 40), (45, 10),
    (20, 25), (5, 30), (35, 20), (15, 15), (30, 45)
]
REWARDS = [
    488, 157, 104, 52, 261,
    314, 209, 401, 122, 348
]
NUM_OUTBREAKS = len(OUTBREAK_POINTS)

# --- Carregamento de Imagens ---
ash_img = None
pokeball_img = None
try:
    ash_img = mpimg.imread('ash.png') 
    pokeball_img = mpimg.imread('pokeball.png')
except FileNotFoundError:
    pass

# --- Pré-cálculo da Matriz de Custos usando A* ---
COSTS_MATRIX = np.zeros((NUM_OUTBREAKS, NUM_OUTBREAKS))
for i in range(NUM_OUTBREAKS):
    for j in range(NUM_OUTBREAKS):
        if i != j:
            start = OUTBREAK_POINTS[i]
            end = OUTBREAK_POINTS[j]
            cost, _ = calcular_custo_a_estrela(TERRAIN_GRID, start, end, TERRAIN_COSTS)
            COSTS_MATRIX[i, j] = cost
        else:
            COSTS_MATRIX[i, j] = 0




# --- Execução do Algoritmo Genético ---
ag = GeneticAlgorithmTSP(OUTBREAK_POINTS, COSTS_MATRIX, REWARDS, population_size=150, generations=300)
best_route_indices, max_fitness = ag.run()

route_reward_sum = sum(REWARDS[i] for i in best_route_indices)
route_cost = route_reward_sum / max_fitness if max_fitness > 0 else float('inf')





# --- Preparar Caminho Detalhado para Animação ---
# O 'simulation_path_cost' armazena o custo acumulado em cada ponto do caminho
full_simulation_path = []
simulation_path_cost = [] 
current_total_cost = 0

for i in range(len(best_route_indices)):
    start_idx = best_route_indices[i]
    end_idx = best_route_indices[(i + 1) % NUM_OUTBREAKS] 
    start_coords = OUTBREAK_POINTS[start_idx]
    end_coords = OUTBREAK_POINTS[end_idx]

    _, segment_path = calcular_custo_a_estrela(TERRAIN_GRID, start_coords, end_coords, TERRAIN_COSTS)

    if segment_path:
        
        # O A* já encontra o caminho ótimo, vamos calcular o custo do segmento passo a passo
        for k in range(len(segment_path)):
            current_r, current_c = segment_path[k]
            
            # Adiciona o ponto atual, mas verifica se é um ponto duplicado de junção de segmentos
            if full_simulation_path and (current_r, current_c) == full_simulation_path[-1]:
                continue
                
            full_simulation_path.append((current_r, current_c))
            
            # Se não é o ponto inicial, adiciona o custo de entrar na célula
            if k > 0 or current_total_cost > 0:
                cost_to_enter = TERRAIN_COSTS.get(TERRAIN_GRID[current_r, current_c], 1)
                current_total_cost += cost_to_enter
            
            simulation_path_cost.append(current_total_cost)

if not full_simulation_path: 
    full_simulation_path.append(OUTBREAK_POINTS[0])
    simulation_path_cost.append(0)


#----------------------------------------------------------------------------------------------------------


# O custo total real da simulação deve ser igual ao route_cost (com alguma pequena variação devido ao arredondamento do float)
# O último valor em simulation_path_cost é o custo total
final_cost_simulated = simulation_path_cost[-1] if simulation_path_cost else 0


# --- Visualização com Animação ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8)) 

# --- Subplot do Mapa de Terreno e Rota (ax1) ---
ax1.imshow(TERRAIN_GRID, cmap=cmap_pokemon, origin='upper', extent=[0, GRID_SIZE, GRID_SIZE, 0])
ax1.set_title('Simulação da Rota Otimizada (Recompensa/Custo)')
ax1.set_xlabel("Coordenada X")
ax1.set_ylabel("Coordenada Y")
ax1.set_xticks(np.arange(0, GRID_SIZE, 5))
ax1.set_yticks(np.arange(0, GRID_SIZE, 5))
ax1.grid(which='major', color='grey', linestyle='-', linewidth=0.5, alpha=0.5)

# --- Legenda do Mapa de Terreno (NOVO) ---
legend_patches = []
for type_id, data in TERRAIN_LEGENDS.items():
    patch = mpatches.Patch(color=data['color'], label=data['name'])
    legend_patches.append(patch)

# Adiciona a legenda na parte inferior do mapa
ax1.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.15),
           ncol=3, title="Tipo de Terreno", fancybox=True, shadow=True,
           fontsize=8)


# Estruturas para o Blit da Animação
animatable_artists = []

# Desenha os outbreaks como imagens ou marcadores
for i, (r, c) in enumerate(OUTBREAK_POINTS):
    text_color = 'red' if REWARDS[i] >= 350 else 'orange' 

    if pokeball_img is not None:
        imagebox = OffsetImage(pokeball_img, zoom=0.25)
        ab = AnnotationBbox(imagebox, (c + 0.5, r + 0.5), frameon=False, zorder=3)
        ax1.add_artist(ab)
        animatable_artists.append(ab)
    else:
        dot, = ax1.plot(c + 0.5, r + 0.5, 'o', color=text_color, markersize=12, markeredgecolor='black', zorder=3)
        animatable_artists.append(dot)
    
    # Texto (que não se move)
    ax1.text(c + 1.2, r + 0.5, f'O{i}(R:{REWARDS[i]})', color='white', fontsize=8, weight='bold',
             bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'), zorder=4)


# Inicializa o Ash/Treinador (Objeto que se move)
if ash_img is not None:
    ash_imagebox = OffsetImage(ash_img, zoom=0.20)
    initial_r, initial_c = full_simulation_path[0]
    pokemon_artist = AnnotationBbox(ash_imagebox, (initial_c + 0.5, initial_r + 0.5), 
                                    frameon=False, zorder=5)
    ax1.add_artist(pokemon_artist)
else:
    pokemon_artist, = ax1.plot([], [], 'P', color='blue', markersize=14, markeredgecolor='white', label='Ash', zorder=5)

# Linha do Caminho Percorrido
path_line, = ax1.plot([], [], 'w-', alpha=0.8, linewidth=3, label='Caminho Percorrido', zorder=2) 

# NOVO: Timer de exibição no mapa
timer_text = ax1.text(0.02, 0.98, f'Custo Atual: 0.00', transform=ax1.transAxes, 
                      fontsize=12, color='white', weight='bold', 
                      bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'), 
                      verticalalignment='top')
animatable_artists.append(timer_text)

animatable_artists.append(pokemon_artist)
animatable_artists.append(path_line)


# --- Subplot da Evolução da Fitness (ax2) ---
ax2.plot(ag.history, color='purple')
ax2.set_title('Evolução da Melhor Fitness por Geração (AG)')
ax2.set_xlabel('Geração')
ax2.set_ylabel('Fitness (Recompensa/Custo)')
ax2.grid(True)
ax2.set_xlim(0, ag.GENERATIONS) 

plt.tight_layout() 


# --- Funções de Animação com Timer ---
def init_animation():
    path_line.set_data([], [])
    timer_text.set_text(f'Custo Atual: 0.00')

    if ash_img is None:
        pokemon_artist.set_data([], [])
    
    return tuple(animatable_artists) 

def update_animation(frame):
    if frame < len(full_simulation_path):
        current_r, current_c = full_simulation_path[frame]
        current_cost = simulation_path_cost[frame] # Custo acumulado






# 1. Atualiza o Timer
        timer_text.set_text(f'Custo Atual: {current_cost:.2f}')
        
# 2. Atualiza a posição do Ash/Pokémon
        if ash_img is not None:
            pokemon_artist.xybox = (current_c + 0.5, current_r + 0.5)
        else:
            pokemon_artist.set_data([current_c + 0.5], [current_r + 0.5]) 

# 3. Atualiza a trilha percorrida
        current_path_x = [c + 0.5 for r, c in full_simulation_path[:frame+1]]
        current_path_y = [r + 0.5 for r, c in full_simulation_path[:frame+1]]
        
        path_line.set_data(current_path_x, current_path_y)
    
    
    
            # Para a animação quando o destino final (último frame) for alcançado

    elif frame == len(full_simulation_path):
        # Frame final da jornada
        pass
    return tuple(animatable_artists) 






# Cria a animação
ani = FuncAnimation(fig, update_animation, frames=len(full_simulation_path) + 1, # Adiciona 1 frame extra para o estado final
                    init_func=init_animation, blit=True, repeat=False, interval=100) 

plt.show()

# --- Tabela de Pontuação (Após a Visualização) ---

print("\n" + "="*50)
print("             🏆 TABELA FINAL DE PONTUAÇÃO 🏆")
print("="*50)
print(f"Rota Otimizada pelo AG: {' -> '.join([f'O{i}' for i in best_route_indices])}")
print("-" * 50)
print(f"| {'Métrica':<25} | {'Valor':<20} |")
print("-" * 50)
print(f"| {'Recompensa Total':<25} | {route_reward_sum:<20.2f} |")
print(f"| {'Custo Ótimo (AG)':<25} | {route_cost:<20.2f} |")
print(f"| {'Custo Simulado (A*)':<25} | {final_cost_simulated:<20.2f} |")
print(f"| {'Fitness (Recompensa/Custo)':<25} | {max_fitness:<20.4f} |")
print("-" * 50)

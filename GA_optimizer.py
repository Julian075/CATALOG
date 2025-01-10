import numpy as np
import random
import json
import argparse
from models import CATALOG_Base as base
from train.Base.Train_CATALOG_Base_out_domain import CATALOG_base
from utils import BaselineDataset,dataloader_baseline,build_optimizer
import time



class GA():
    def __init__(self,HYPERPARAM_RANGES,ruta_features_train,ruta_features_val,ruta_features_test1,ruta_features_test2,path_text_feat1,path_text_feat2):
        self.HYPERPARAM_RANGES=HYPERPARAM_RANGES
        self.ruta_features_train = ruta_features_train
        self.ruta_features_val   = ruta_features_val
        self.ruta_features_test1 = ruta_features_test1
        self.ruta_features_test2 = ruta_features_test2
        self.path_text_feat1     = path_text_feat1
        self.path_text_feat2     = path_text_feat2


    def objective_function(self,individual,exp_name):
        """
           Evalúa el modelo con un conjunto de hiperparámetros y devuelve la puntuación.
           """
        model = CATALOG_base(weight_Clip=individual['weight_Clip'], num_epochs=individual['num_epochs'], batch_size=individual['batch_size'], num_layers=individual['num_layers'],
                             dropout=individual['dropout'], hidden_dim=individual['hidden_dim'], lr=individual['lr'], t=individual['t'], momentum=individual['momentum']
                             , patience=5, model=base, Dataset=BaselineDataset,
                             Dataloader=dataloader_baseline, version='base', ruta_features_train=self.ruta_features_train,
                             ruta_features_val=self.ruta_features_val, ruta_features_test1=self.ruta_features_test1,
                             ruta_features_test2=self.ruta_features_test2, path_text_feat1=self.path_text_feat1,
                             path_text_feat2=self.path_text_feat2, build_optimizer=build_optimizer,
                             exp_name=exp_name)
        validation_acc= model.train_HPO()

        return validation_acc
    def test(self,individual,model_params_path):
        """
           Prueba el modelo con un conjunto de hiperparámetros y devuelve la puntuación.
           """
        model = CATALOG_base(weight_Clip=individual['weight_Clip'], num_epochs=individual['num_epochs'], batch_size=individual['batch_size'], num_layers=individual['num_layers'],
                             dropout=individual['dropout'], hidden_dim=individual['hidden_dim'], lr=individual['lr'], t=individual['t'], momentum=individual['momentum']
                             , patience=5, model=base, Dataset=BaselineDataset,
                             Dataloader=dataloader_baseline, version='base', ruta_features_train=self.ruta_features_train,
                             ruta_features_val=self.ruta_features_val, ruta_features_test1=self.ruta_features_test1,
                             ruta_features_test2=self.ruta_features_test2, path_text_feat1=self.path_text_feat1,
                             path_text_feat2=self.path_text_feat2, build_optimizer=build_optimizer,
                             exp_name="NA")
        model.prueba_model(model_params_path)

    # Genera un conjunto inicial de soluciones aleatorias
    def initialize_population(self,size):
        population = []
        for _ in range(size):
            individual = {key: random.uniform(*value) if isinstance(value, tuple) else random.choice(value)
                          for key, value in HYPERPARAM_RANGES.items()}
            individual['num_epochs'] = int(individual['num_epochs'])  # Garantizamos que num_epochs sea entero
            individual['batch_size'] = int(individual['batch_size'])  # Garantizamos que batch_size sea entero
            individual['num_layers'] = int(individual['num_layers'])  # Garantizamos que num_layers sea entero
            individual['hidden_dim'] = int(individual['hidden_dim'])  # Garantizamos que hidden_dim sea entero
            population.append(individual)
        return population


    # Genera soluciones combinando dos padres
    def combine_solutions(self,parent1, parent2):
        child = {}
        for key in parent1:
            if np.random.rand() > 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child


    # Muta un individuo ligeramente
    def mutate_solution(self,individual, mutation_rate=0.1):
        mutated = individual.copy()
        for key in HYPERPARAM_RANGES:
            if random.random() < mutation_rate:
                range_min, range_max = HYPERPARAM_RANGES[key]
                mutation = random.uniform(-0.1, 0.1) * (range_max - range_min)
                mutated[key] = max(range_min, min(range_max, mutated[key] + mutation))
                if key in ['num_epochs', 'batch_size', 'num_layers', 'hidden_dim']:
                    mutated[key] = int(mutated[key])  # Garantizamos que los valores discretos permanezcan como enteros
        return mutated


    # Scatter Search
    def GA(self,population_size=10, generations=20, elite_size=5, mutation_rate=0.1):
        ti=time.time()
        # Genera la población inicial
        population = self.initialize_population(population_size)
        best_ind_gen = []

        for generation in range(generations):
            print(f"Generation {generation + 1}/{generations}")
            # Evaluamos la población
            population_scores=[]
            counter=0
            for individual in population:
                counter += 1  # Incrementar el contador
                exp_name =f"{generation}_{counter}"  # Crear exp_name único
                score = self.objective_function(individual, exp_name)  # Evaluar individuo
                population_scores.append((individual, score))  # Agregar a la lista
            population_scores.sort(key=lambda x: x[1][0], reverse=True)  # Ordenamos por puntuación (mayor es mejor)

            # Mantenemos a los mejores individuos
            elite = [ind[0] for ind in population_scores[:elite_size]]
            best_ind_gen.append({
                "generation": generation + 1,
                "best_individual": population_scores[0][0],  # Parámetros del mejor individuo
                "best_score": population_scores[0][1][0],  # Mejor puntuación
                "model_path": population_scores[0][1][1],  # Ruta al modelo guardado
            })
            #best_ind_gen.append(population_scores[0]) # (dict with parameters, list [ acc_value , path_model] )
            # Guardar resultados en un archivo JSON
            with open("GA_best_ind_gen.json", "w") as json_file:
                json.dump(best_ind_gen, json_file, indent=4)

            # Generamos nuevos individuos
            new_population = elite.copy()
            while len(new_population) < population_size:
                # Combinamos dos padres aleatorios de los mejores
                parent1, parent2 = random.sample(elite, 2)
                child = self.combine_solutions(parent1, parent2)
                child = self.mutate_solution(child, mutation_rate)
                new_population.append(child)

            population = new_population

        # Devolvemos el mejor conjunto de hiperparámetros encontrado
        population_scores.sort(key=lambda x: x[1][0], reverse=True)  # Ordenamos por puntuación (mayor es mejor)
        best_individual = population_scores[0][0]
        best_score = population_scores[0][1][0]
        model_params_path= population_scores[0][1][1]

        print(f"Best individual: {best_individual}")
        print(f"Best score: {best_score}")

        # Guardar resultados en un archivo JSON
        with open("GA_best_ind_gen.json", "w") as json_file:
            json.dump(best_ind_gen, json_file, indent=4)

        tf = time.time()
        tf = tf - ti
        print(f" tiempo total: {tf}")

        #print results for test data
        self.test(best_individual, model_params_path)
        return best_individual, best_score



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GA Search for Hyperparameter Optimization")
    parser.add_argument("--population_size", type=int, default=10, help="Size of the population for GA search")
    parser.add_argument("--generations", type=int, default=20, help="Number of generations for GA search")
    parser.add_argument("--elite_size", type=int, default=5, help="Size of the elite set")
    parser.add_argument("--mutation_rate", type=float, default=0.1, help="Mutation rate for the search")

    args = parser.parse_args()

    # Definimos el rango de hiperparámetros
    HYPERPARAM_RANGES = {
        'weight_Clip': (0.4, 0.7),  # Ajustado alrededor de 0.6, que dio buenos resultados
        'num_epochs': (5, 20),  # Reducido, ya que 8 épocas funcionaron bien
        'batch_size': (32, 64),  # Ajustado alrededor de 48, favoreciendo lotes pequeños
        'num_layers': (1, 5),  # Enfoque en arquitecturas más simples con pocas capas
        'dropout': (0.2, 0.5),  # Muy cerca de 0.27 para afinar
        'hidden_dim': (900, 1200),  # Ajustado alrededor de 1045
        'lr': (0.05, 0.1),  # Alrededor de 0.08, evitando tasas muy altas o bajas
        't': (0.0, 1.0),  # Fijo, ya que 0.1 funcionó
        'momentum': (0.7, 0.9),  # Ajustado cerca de 0.8
    }

    ruta_features_train = "features/Features_serengeti/standard_features/Features_CATALOG_train_16.pt"
    ruta_features_val = "features/Features_serengeti/standard_features/Features_CATALOG_val_16.pt"
    ruta_features_test1 = "features/Features_terra/standard_features/Features_CATALOG_cis_test_16.pt"
    ruta_features_test2 = "features/Features_terra/standard_features/Features_CATALOG_trans_test_16.pt"
    path_text_feat1 = "features/Features_serengeti/standard_features/Text_features_16.pt"
    path_text_feat2 = "features/Features_terra/standard_features/Text_features_16.pt"


    HPO_model = GA(
        HYPERPARAM_RANGES,
        ruta_features_train=ruta_features_train,
        ruta_features_val=ruta_features_val,
        ruta_features_test1=ruta_features_test1,
        ruta_features_test2=ruta_features_test2,
        path_text_feat1=path_text_feat1,
        path_text_feat2=path_text_feat2
    )

    best_hyperparams, best_score = HPO_model.GA(
        population_size=args.population_size,
        generations=args.generations,
        elite_size=args.elite_size,
        mutation_rate=args.mutation_rate
    )

    print("Mejores hiperparámetros encontrados:")
    print(best_hyperparams)
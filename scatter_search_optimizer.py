import time
import random
import json
import argparse
from models import CATALOG_Base as base
from train.Base.Train_CATALOG_Base_out_domain import CATALOG_base
from utils import BaselineDataset, dataloader_baseline, build_optimizer


class scatter_search():
    def __init__(self, HYPERPARAM_RANGES, ruta_features_train, ruta_features_val, ruta_features_test1, ruta_features_test2, path_text_feat1, path_text_feat2):
        self.HYPERPARAM_RANGES = HYPERPARAM_RANGES
        self.ruta_features_train = ruta_features_train
        self.ruta_features_val = ruta_features_val
        self.ruta_features_test1 = ruta_features_test1
        self.ruta_features_test2 = ruta_features_test2
        self.path_text_feat1 = path_text_feat1
        self.path_text_feat2 = path_text_feat2

    def objective_function(self, individual, exp_name):
        """
        Evalúa el modelo con un conjunto de hiperparámetros y devuelve la puntuación.
        """
        model = CATALOG_base(
            weight_Clip=individual['weight_Clip'], num_epochs=individual['num_epochs'],
            batch_size=individual['batch_size'], num_layers=individual['num_layers'],
            dropout=individual['dropout'], hidden_dim=individual['hidden_dim'],
            lr=individual['lr'], t=individual['t'], momentum=individual['momentum'],
            patience=5, model=base, Dataset=BaselineDataset,
            Dataloader=dataloader_baseline, version='base',
            ruta_features_train=self.ruta_features_train,
            ruta_features_val=self.ruta_features_val,
            ruta_features_test1=self.ruta_features_test1,
            ruta_features_test2=self.ruta_features_test2,
            path_text_feat1=self.path_text_feat1, path_text_feat2=self.path_text_feat2,
            build_optimizer=build_optimizer, exp_name=exp_name
        )
        validation_acc = model.train_HPO()
        return validation_acc

    def test(self, individual, model_params_path):
        """
        Prueba el modelo con un conjunto de hiperparámetros y devuelve la puntuación.
        """
        model = CATALOG_base(
            weight_Clip=individual['weight_Clip'], num_epochs=individual['num_epochs'],
            batch_size=individual['batch_size'], num_layers=individual['num_layers'],
            dropout=individual['dropout'], hidden_dim=individual['hidden_dim'],
            lr=individual['lr'], t=individual['t'], momentum=individual['momentum'],
            patience=5, model=base, Dataset=BaselineDataset,
            Dataloader=dataloader_baseline, version='base',
            ruta_features_train=self.ruta_features_train,
            ruta_features_val=self.ruta_features_val,
            ruta_features_test1=self.ruta_features_test1,
            ruta_features_test2=self.ruta_features_test2,
            path_text_feat1=self.path_text_feat1, path_text_feat2=self.path_text_feat2,
            build_optimizer=build_optimizer, exp_name="NA"
        )
        model.prueba_model(model_params_path)

    def initialize_population(self, size):
        population = []
        for _ in range(size):
            individual = {key: random.uniform(*value) if isinstance(value, tuple) else random.choice(value)
                          for key, value in self.HYPERPARAM_RANGES.items()}
            individual['num_epochs'] = int(individual['num_epochs'])
            individual['batch_size'] = int(individual['batch_size'])
            individual['num_layers'] = int(individual['num_layers'])
            individual['hidden_dim'] = int(individual['hidden_dim'])
            population.append(individual)
        return population

    def combine_solutions(self, parent1, parent2):
        # Combinación estructurada: promedio ponderado de los valores
        alpha = random.uniform(0.4, 0.6)  # Ponderación aleatoria
        combined = {key: alpha * parent1[key] + (1 - alpha) * parent2[key] for key in parent1.keys()}
        return combined

    def local_improvement(self, individual, max_attempts=5):
        """
        Mejora local basada en ajustes pequeños a los parámetros de la solución.
        """
        improved = individual.copy()  # Copia para no modificar la solución original
        best_score = self.objective_function(improved, exp_name="local_improvement")
        attempts = 0  # Contador de intentos fallidos

        while attempts < max_attempts:
            param_to_modify = random.choice(list(self.HYPERPARAM_RANGES.keys()))
            if isinstance(self.HYPERPARAM_RANGES[param_to_modify], tuple):
                step = random.uniform(-0.05, 0.05) * (self.HYPERPARAM_RANGES[param_to_modify][1] - self.HYPERPARAM_RANGES[param_to_modify][0])
                new_value = improved[param_to_modify] + step
                new_value = min(max(new_value, self.HYPERPARAM_RANGES[param_to_modify][0]), self.HYPERPARAM_RANGES[param_to_modify][1])
                improved[param_to_modify] = new_value

            new_score = self.objective_function(improved, exp_name="local_improvement_test")
            if new_score > best_score:
                best_score = new_score
                attempts = 0
            else:
                improved[param_to_modify] = individual[param_to_modify]
                attempts += 1

        return improved

    def scatter_search(self, population_size=10, generations=20, ref_set_size=5):
        ti=time.time()
        # Genera la población inicial
        population = self.initialize_population(population_size)
        reference_set = []  # Conjunto de referencia
        best_ind_gen = []

        for generation in range(generations):
            print(f"Generation {generation + 1}/{generations}")

            population_scores = [(ind, self.objective_function(ind, f"{generation}_{i}")) for i, ind in enumerate(population)]
            population_scores.sort(key=lambda x: x[1], reverse=True)  # Ordenar por puntuación descendente
            reference_set = population_scores[:ref_set_size]  # Tomar los mejores como referencia

            new_population = []
            for i in range(len(reference_set)):
                for j in range(i + 1, len(reference_set)):
                    child = self.combine_solutions(reference_set[i][0], reference_set[j][0])
                    child = self.local_improvement(child)
                    new_population.append(child)

            population = new_population
            best_ind_gen.append({
                "generation": generation + 1,
                "best_individual": reference_set[0][0],
                "best_score": reference_set[0][1],
            })
            with open("SCS_best_ind_gen.json", "w") as json_file:
                json.dump(best_ind_gen, json_file, indent=4)

        best_individual = reference_set[0][0]
        best_score = reference_set[0][1]

        print(f"Best individual: {best_individual}")
        print(f"Best score: {best_score}")

        with open("SCS_best_ind_gen.json", "w") as json_file:
            json.dump(best_ind_gen, json_file, indent=4)

        tf=time.time()
        tf=tf-ti
        print(f" tiempo total: {tf}")

        return best_individual, best_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scatter Search for Hyperparameter Optimization")
    parser.add_argument("--population_size", type=int, default=10, help="Size of the population for scatter search")
    parser.add_argument("--generations", type=int, default=20, help="Number of generations for scatter search")
    parser.add_argument("--ref_set_size", type=int, default=5, help="Size of the reference set")

    args = parser.parse_args()

    HYPERPARAM_RANGES = {
        'weight_Clip': (0.4, 0.7),
        'num_epochs': (5, 20),
        'batch_size': (32, 64),
        'num_layers': (1, 5),
        'dropout': (0.2, 0.5),
        'hidden_dim': (900, 1200),
        'lr': (0.05, 0.1),
        't': (0.0, 1.0),
        'momentum': (0.7, 0.9),
    }

    ruta_features_train = "features/Features_serengeti/standard_features/Features_CATALOG_train_16.pt"
    ruta_features_val = "features/Features_serengeti/standard_features/Features_CATALOG_val_16.pt"
    ruta_features_test1 = "features/Features_terra/standard_features/Features_CATALOG_cis_test_16.pt"
    ruta_features_test2 = "features/Features_terra/standard_features/Features_CATALOG_trans_test_16.pt"
    path_text_feat1 = "features/Features_serengeti/standard_features/Text_features_16.pt"
    path_text_feat2 = "features/Features_terra/standard_features/Text_features_16.pt"

    HPO_model = scatter_search(
        HYPERPARAM_RANGES,
        ruta_features_train=ruta_features_train,
        ruta_features_val=ruta_features_val,
        ruta_features_test1=ruta_features_test1,
        ruta_features_test2=ruta_features_test2,
        path_text_feat1=path_text_feat1,
        path_text_feat2=path_text_feat2
    )

    best_hyperparams, best_score = HPO_model.scatter_search(
        population_size=args.population_size,
        generations=args.generations,
        ref_set_size=args.ref_set_size
    )

    print("Mejores hiperparámetros encontrados:")
    print(best_hyperparams)

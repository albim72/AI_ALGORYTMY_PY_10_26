"""
Algorytm mrówkowy do doboru cech w regresji.

Przykład używa zbioru diabetes z biblioteki scikit-learn.
Każda mrówka wybiera podzbiór cech.
Jakość podzbioru oceniamy przez walidację krzyżową modelu Ridge Regression.

Wymagania:
    pip install numpy scikit-learn
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


@dataclass
class AntResult:
    """Wynik pojedynczej mrówki."""
    selected_features: np.ndarray
    score: float
    fitness: float


class AntColonyFeatureSelector:
    """
    Algorytm mrówkowy do wyboru cech.

    Każda cecha ma swój poziom feromonu.
    Im więcej feromonu ma cecha, tym chętniej jest wybierana przez kolejne mrówki.
    """

    def __init__(
        self,
        n_ants: int = 30,
        n_iterations: int = 40,
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation_rate: float = 0.3,
        q: float = 1.0,
        min_features: int = 2,
        max_features: Optional[int] = None,
        penalty: float = 0.02,
        random_state: int = 42,
    ) -> None:
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.q = q
        self.min_features = min_features
        self.max_features = max_features
        self.penalty = penalty
        self.random_state = random_state

        self.best_features_: Optional[np.ndarray] = None
        self.best_score_: float = -np.inf
        self.best_fitness_: float = -np.inf
        self.pheromone_: Optional[np.ndarray] = None

    def _build_heuristic(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Tworzy heurystykę atrakcyjności cech.

        W tym przykładzie używamy bezwzględnej korelacji cechy z targetem.
        Im większa korelacja, tym cecha jest bardziej kusząca dla mrówek.
        """
        heuristic = []

        for feature_index in range(x.shape[1]):
            correlation_matrix = np.corrcoef(x[:, feature_index], y)
            correlation = correlation_matrix[0, 1]

            if np.isnan(correlation):
                correlation = 0.0

            heuristic.append(abs(correlation))

        heuristic = np.array(heuristic)
        heuristic = heuristic + 1e-8

        return heuristic / heuristic.sum()

    def _select_features(
        self,
        pheromone: np.ndarray,
        heuristic: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Jedna mrówka wybiera podzbiór cech.

        Prawdopodobieństwo wyboru cechy zależy od:
        - feromonu,
        - heurystyki,
        - parametrów alpha i beta.
        """
        n_features = len(pheromone)

        max_features = self.max_features
        if max_features is None:
            max_features = n_features

        subset_size = rng.integers(self.min_features, max_features + 1)

        attractiveness = (pheromone ** self.alpha) * (heuristic ** self.beta)
        probabilities = attractiveness / attractiveness.sum()

        selected = rng.choice(
            np.arange(n_features),
            size=subset_size,
            replace=False,
            p=probabilities,
        )

        return np.sort(selected)

    def _evaluate_subset(
        self,
        x: np.ndarray,
        y: np.ndarray,
        selected_features: np.ndarray,
    ) -> AntResult:
        """
        Ocenia podzbiór cech.

        score:
            Średni wynik R2 z walidacji krzyżowej.

        fitness:
            Wynik z karą za zbyt dużą liczbę cech.
        """
        model = Ridge(alpha=1.0)

        scores = cross_val_score(
            model,
            x[:, selected_features],
            y,
            cv=5,
            scoring="r2",
        )

        score = float(scores.mean())

        feature_ratio = len(selected_features) / x.shape[1]
        fitness = score - self.penalty * feature_ratio

        return AntResult(
            selected_features=selected_features,
            score=score,
            fitness=fitness,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> "AntColonyFeatureSelector":
        """Uruchamia algorytm mrówkowy."""
        rng = np.random.default_rng(self.random_state)

        n_features = x.shape[1]
        pheromone = np.ones(n_features)
        heuristic = self._build_heuristic(x, y)

        for iteration in range(1, self.n_iterations + 1):
            ant_results = []

            for _ in range(self.n_ants):
                selected_features = self._select_features(
                    pheromone=pheromone,
                    heuristic=heuristic,
                    rng=rng,
                )

                result = self._evaluate_subset(x, y, selected_features)
                ant_results.append(result)

                if result.fitness > self.best_fitness_:
                    self.best_fitness_ = result.fitness
                    self.best_score_ = result.score
                    self.best_features_ = result.selected_features

            pheromone *= 1.0 - self.evaporation_rate

            fitness_values = np.array([result.fitness for result in ant_results])
            min_fitness = fitness_values.min()

            for result in ant_results:
                positive_contribution = result.fitness - min_fitness + 1e-8
                pheromone[result.selected_features] += self.q * positive_contribution

            pheromone = np.clip(pheromone, 1e-6, 100.0)

            if iteration % 5 == 0:
                print(
                    f"Iteracja {iteration:2d} | "
                    f"najlepsze R2: {self.best_score_:.4f} | "
                    f"liczba cech: {len(self.best_features_)} | "
                    f"cechy: {self.best_features_}"
                )

        self.pheromone_ = pheromone

        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Zwraca dane ograniczone do najlepszych znalezionych cech."""
        if self.best_features_ is None:
            raise RuntimeError("Najpierw uruchom metodę fit().")

        return x[:, self.best_features_]

    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Uruchamia fit i od razu zwraca wybrane cechy."""
        self.fit(x, y)
        return self.transform(x)


if __name__ == "__main__":
    data = load_diabetes()

    x = data.data
    y = data.target
    feature_names = np.array(data.feature_names)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    selector = AntColonyFeatureSelector(
        n_ants=30,
        n_iterations=40,
        alpha=1.0,
        beta=2.0,
        evaporation_rate=0.3,
        q=1.0,
        min_features=2,
        max_features=8,
        penalty=0.02,
        random_state=42,
    )

    selector.fit(x_scaled, y)

    selected_names = feature_names[selector.best_features_]

    print("\nNajlepszy znaleziony podzbiór cech:")
    print(selected_names)

    print("\nIndeksy cech:")
    print(selector.best_features_)

    print("\nNajlepszy wynik R2:")
    print(round(selector.best_score_, 4))

    print("\nNajlepsza wartość fitness:")
    print(round(selector.best_fitness_, 4))

    print("\nKońcowy poziom feromonu dla cech:")
    for name, pheromone_value in zip(feature_names, selector.pheromone_):
        print(f"{name:>5s}: {pheromone_value:.4f}")

"""
Algorytm mrówkowy dla problemu komiwojażera, TSP.

Autor: przykład szkoleniowy
Wymagania: czysty Python, bez bibliotek zewnętrznych
"""

import math
import random
from typing import List, Tuple


City = Tuple[float, float]


def euclidean_distance(city_a: City, city_b: City) -> float:
    """Oblicza odległość euklidesową między dwoma miastami."""
    return math.sqrt((city_a[0] - city_b[0]) ** 2 + (city_a[1] - city_b[1]) ** 2)


def build_distance_matrix(cities: List[City]) -> List[List[float]]:
    """Tworzy macierz odległości między wszystkimi miastami."""
    n = len(cities)
    distances = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j:
                distances[i][j] = euclidean_distance(cities[i], cities[j])

    return distances


def calculate_tour_length(tour: List[int], distances: List[List[float]]) -> float:
    """Oblicza całkowitą długość trasy, wraz z powrotem do miasta startowego."""
    total = 0.0

    for i in range(len(tour) - 1):
        total += distances[tour[i]][tour[i + 1]]

    total += distances[tour[-1]][tour[0]]

    return total


def choose_next_city(
    current_city: int,
    unvisited: List[int],
    pheromone: List[List[float]],
    distances: List[List[float]],
    alpha: float,
    beta: float,
) -> int:
    """
    Wybiera następne miasto na podstawie feromonu i heurystyki.

    Feromon mówi: tam wcześniej były dobre trasy.
    Heurystyka mówi: bliższe miasta są bardziej atrakcyjne.
    """
    probabilities = []
    denominator = 0.0

    for city in unvisited:
        tau = pheromone[current_city][city] ** alpha
        eta = (1.0 / distances[current_city][city]) ** beta
        value = tau * eta

        probabilities.append((city, value))
        denominator += value

    if denominator == 0:
        return random.choice(unvisited)

    random_value = random.random()
    cumulative = 0.0

    for city, value in probabilities:
        probability = value / denominator
        cumulative += probability

        if random_value <= cumulative:
            return city

    return unvisited[-1]


def construct_tour(
    start_city: int,
    pheromone: List[List[float]],
    distances: List[List[float]],
    alpha: float,
    beta: float,
) -> List[int]:
    """Buduje jedną trasę jednej mrówki."""
    n = len(distances)
    tour = [start_city]
    unvisited = list(range(n))
    unvisited.remove(start_city)

    while unvisited:
        current_city = tour[-1]
        next_city = choose_next_city(
            current_city=current_city,
            unvisited=unvisited,
            pheromone=pheromone,
            distances=distances,
            alpha=alpha,
            beta=beta,
        )

        tour.append(next_city)
        unvisited.remove(next_city)

    return tour


def evaporate_pheromone(
    pheromone: List[List[float]],
    evaporation_rate: float,
) -> None:
    """Zmniejsza ilość feromonu na wszystkich ścieżkach."""
    n = len(pheromone)

    for i in range(n):
        for j in range(n):
            pheromone[i][j] *= 1.0 - evaporation_rate


def deposit_pheromone(
    pheromone: List[List[float]],
    tours: List[List[int]],
    distances: List[List[float]],
    q: float,
) -> None:
    """
    Wzmacnia feromon na krawędziach dobrych tras.

    Krótsza trasa dostaje większe wzmocnienie.
    """
    for tour in tours:
        tour_length = calculate_tour_length(tour, distances)
        contribution = q / tour_length

        for i in range(len(tour) - 1):
            a = tour[i]
            b = tour[i + 1]
            pheromone[a][b] += contribution
            pheromone[b][a] += contribution

        last = tour[-1]
        first = tour[0]
        pheromone[last][first] += contribution
        pheromone[first][last] += contribution


def ant_colony_tsp(
    cities: List[City],
    n_ants: int = 20,
    n_iterations: int = 100,
    alpha: float = 1.0,
    beta: float = 3.0,
    evaporation_rate: float = 0.4,
    q: float = 100.0,
    seed: int = 42,
) -> Tuple[List[int], float]:
    """Główna funkcja rozwiązująca problem TSP algorytmem mrówkowym."""
    random.seed(seed)

    n = len(cities)
    distances = build_distance_matrix(cities)

    pheromone = [[1.0 for _ in range(n)] for _ in range(n)]

    best_tour = []
    best_length = float("inf")

    for iteration in range(1, n_iterations + 1):
        tours = []

        for _ in range(n_ants):
            start_city = random.randint(0, n - 1)
            tour = construct_tour(
                start_city=start_city,
                pheromone=pheromone,
                distances=distances,
                alpha=alpha,
                beta=beta,
            )

            tours.append(tour)

            length = calculate_tour_length(tour, distances)

            if length < best_length:
                best_length = length
                best_tour = tour

        evaporate_pheromone(pheromone, evaporation_rate)
        deposit_pheromone(pheromone, tours, distances, q)

        if iteration % 10 == 0:
            print(f"Iteracja {iteration:3d} | najlepsza długość: {best_length:.2f}")

    return best_tour, best_length


if __name__ == "__main__":
    cities = [
        (0, 0),
        (2, 6),
        (3, 1),
        (5, 5),
        (6, 2),
        (8, 3),
        (7, 7),
        (1, 4),
        (4, 7),
        (9, 6),
    ]

    best_tour, best_length = ant_colony_tsp(cities)

    print("\nNajlepsza znaleziona trasa:")
    print(best_tour)

    print("\nDługość trasy:")
    print(round(best_length, 2))

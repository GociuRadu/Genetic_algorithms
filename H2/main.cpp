#include <iostream>
#include <vector>
#include <bitset>
#include <random>
#include <algorithm>
#include <cmath>
#include <string>
#include <iomanip>
#include <numeric>

using namespace std;

#define PI 3.1415926535897

const int POPULATION_SIZE = 100;
const int CHROMOSOME_LENGTH = 20;
const int GENERATIONS = 2000;
const double MUTATION_RATE = 0.007;
const int TOURNAMENT_SIZE = 20;
const int DIMENSIONS = 10;
const int ELITISM_COUNT = 20;

double deJong(const vector<double>& values) {
    double result = 0.0;
    for (double x : values) {
        result += x * x;
    }
    return result;
}

double schwefel(const vector<double>& values) {
    double result = 0.0;
    for (double x : values) {
        result += -x * sin(sqrt(abs(x)));
    }
    return result;
}

double rastrigin(const vector<double>& values) {
    double result = 10.0 * values.size();
    for (double x : values) {
        result += (x * x - 10.0 * cos(2.0 * PI * x));
    }
    return result;
}

double michalewicz(const vector<double>& values, double m = 10.0) {
    double result = 0.0;
    for (size_t i = 0; i < values.size(); ++i) {
        result -= sin(values[i]) * pow(sin((i + 1) * values[i] * values[i] / PI), 2 * m);
    }
    return result;
}

double binaryToReal(const bitset<CHROMOSOME_LENGTH>& binary, double lowerBound, double upperBound) {
    int value = static_cast<int>(binary.to_ulong());
    int maxValue = (1 << CHROMOSOME_LENGTH) - 1;
    return lowerBound + (value / static_cast<double>(maxValue)) * (upperBound - lowerBound);
}

vector<vector<bitset<CHROMOSOME_LENGTH>>> generateInitialPopulation(int populationSize, mt19937& rng) {
    vector<vector<bitset<CHROMOSOME_LENGTH>>> population;
    uniform_int_distribution<int> bitDist(0, 1);
    for (int i = 0; i < populationSize; ++i) {
        vector<bitset<CHROMOSOME_LENGTH>> individual;
        for (int j = 0; j < DIMENSIONS; ++j) {
            bitset<CHROMOSOME_LENGTH> chromosome;
            for (size_t k = 0; k < chromosome.size(); ++k) {
                chromosome[k] = bitDist(rng);
            }
            individual.push_back(chromosome);
        }
        population.push_back(individual);
    }
    return population;
}

vector<double> decodeSolution(const vector<bitset<CHROMOSOME_LENGTH>>& individual, double lowerBound, double upperBound) {
    vector<double> values;
    for (const auto& chromosome : individual) {
        values.push_back(binaryToReal(chromosome, lowerBound, upperBound));
    }
    return values;
}

vector<double> evaluatePopulation(const vector<vector<bitset<CHROMOSOME_LENGTH>>>& population,
    const string& functionName,
    double lowerBound,
    double upperBound) {
    vector<double> fitness;
    for (const auto& individual : population) {
        vector<double> realValues = decodeSolution(individual, lowerBound, upperBound);
        if (functionName == "DeJong") {
            fitness.push_back(deJong(realValues));
        }
        else if (functionName == "Schwefel") {
            fitness.push_back(schwefel(realValues));
        }
        else if (functionName == "Rastrigin") {
            fitness.push_back(rastrigin(realValues));
        }
        else if (functionName == "Michalewicz") {
            fitness.push_back(michalewicz(realValues));
        }
    }
    return fitness;
}

int tournamentSelection(const vector<double>& fitness, mt19937& rng) {
    uniform_int_distribution<int> dist(0, POPULATION_SIZE - 1);
    int best = dist(rng);
    for (int i = 1; i < TOURNAMENT_SIZE; ++i) {
        int competitor = dist(rng);
        if (fitness[competitor] < fitness[best]) {
            best = competitor;
        }
    }
    return best;
}

vector<bitset<CHROMOSOME_LENGTH>> crossover(const vector<bitset<CHROMOSOME_LENGTH>>& parent1,
    const vector<bitset<CHROMOSOME_LENGTH>>& parent2, mt19937& rng) {
    uniform_int_distribution<int> dist(0, CHROMOSOME_LENGTH - 1);
    int crossoverPoint = dist(rng);
    vector<bitset<CHROMOSOME_LENGTH>> child(DIMENSIONS);
    for (int i = 0; i < DIMENSIONS; ++i) {
        for (int j = 0; j < CHROMOSOME_LENGTH; ++j) {
            if (j < crossoverPoint) {
                child[i][j] = parent1[i][j];
            }
            else {
                child[i][j] = parent2[i][j];
            }
        }
    }
    return child;
}

void mutate(vector<bitset<CHROMOSOME_LENGTH>>& individual, double mutationRate, mt19937& rng) {
    uniform_real_distribution<double> dist(0.0, 1.0);
    for (auto& chromosome : individual) {
        for (size_t i = 0; i < chromosome.size(); ++i) {
            if (dist(rng) < mutationRate) {
                chromosome.flip(i);
            }
        }
    }
}

void geneticAlgorithm(const string& functionName, double lowerBound, double upperBound) {
    mt19937 rng(random_device{}());
    vector<double> allFitnessResults;
    cout << fixed << setprecision(5); 
    for (int run = 0; run < 5; ++run) {
        auto population = generateInitialPopulation(POPULATION_SIZE, rng);
        double globalBestFitness = numeric_limits<double>::max();
        vector<double> globalBestSolution;
        for (int generation = 0; generation < GENERATIONS; ++generation) {
            auto fitness = evaluatePopulation(population, functionName, lowerBound, upperBound);
            for (int i = 0; i < POPULATION_SIZE; ++i) {
                if (fitness[i] < globalBestFitness) {
                    globalBestFitness = fitness[i];
                    globalBestSolution = decodeSolution(population[i], lowerBound, upperBound);
                }
            }
            vector<vector<bitset<CHROMOSOME_LENGTH>>> newPopulation;
            for (int i = 0; i < ELITISM_COUNT; ++i) {
                newPopulation.push_back(population[i]);
            }
            while (newPopulation.size() < POPULATION_SIZE) {
                int parent1Index = tournamentSelection(fitness, rng);
                int parent2Index = tournamentSelection(fitness, rng);
                auto child = crossover(population[parent1Index], population[parent2Index], rng);
                mutate(child, MUTATION_RATE, rng);
                newPopulation.push_back(child);
            }
            population = newPopulation;
        }
        allFitnessResults.push_back(globalBestFitness);
    }
    double mean = accumulate(allFitnessResults.begin(), allFitnessResults.end(), 0.0) / allFitnessResults.size();
    double min = *min_element(allFitnessResults.begin(), allFitnessResults.end());
    double max = *max_element(allFitnessResults.begin(), allFitnessResults.end());
    double variance = 0.0;
    for (double fitness : allFitnessResults) {
        variance += (fitness - mean) * (fitness - mean);
    }
    double stddev = sqrt(variance / allFitnessResults.size());
    cout << functionName << " - Media: " << mean << ", Min: " << min << ", Max: " << max << ", StdDev: " << stddev << endl;
}

int main() {
    vector<string> functions = { "Rastrigin" };
    vector<pair<double, double>> bounds = { {-5.12, 5.12} };
    for (size_t i = 0; i < functions.size(); ++i) {
        geneticAlgorithm(functions[i], bounds[i].first, bounds[i].second);
    }
    return 0;
}
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>
#include <numeric>
#include <algorithm>

#define PI 3.1415926535

#define RASTRIGIN_LOWER_BOUND -5.12
#define RASTRIGIN_UPPER_BOUND 5.12

#define DEJONG_LOWER_BOUND -5.12
#define DEJONG_UPPER_BOUND 5.12

#define SCHWEFEL_LOWER_BOUND -500.0
#define SCHWEFEL_UPPER_BOUND 500.0

#define MICHALEWICZ_LOWER_BOUND 0.0
#define MICHALEWICZ_UPPER_BOUND PI

#define STEP_SIZE 0.1
#define MAX_ITERATIONS 10000

int DIMENSION = 10;
const int RUNS = 30;

double rastrigin(const std::vector<double>& x) {
    double sum = 0.0;
    for (double val : x)
        sum += val * val - 10 * std::cos(2 * PI * val);
    return 10 * x.size() + sum;
}

double dejong(const std::vector<double>& x) 
{
    double sum = 0.0;
    for (double val : x)
        sum += val * val;
    return sum;
}
double schwefel(const std::vector<double>& x) 
{
    double sum = 0.0;
    for (double val : x)
        sum += -val * std::sin(std::sqrt(std::abs(val)));
    return sum;
}

double michalewicz(const std::vector<double>& x) {
    double sum = 0.0;
    for (int i = 0; i < x.size(); ++i)
        sum += std::sin(x[i]) * std::pow(std::sin((i + 1) * x[i] * x[i] / PI), 20);
    return -sum;
}

std::vector<double> generateRandomSolution(double lower_bound, double upper_bound, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(lower_bound, upper_bound);
    std::vector<double> solution(DIMENSION);
    for (int i = 0; i < DIMENSION; ++i)
        solution[i] = dist(rng);
    return solution;
}

double calculateMean(const std::vector<double>& values) {
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / values.size();
}

double calculateStandardDeviation(const std::vector<double>& values, double mean) 
{
    double sq_sum = 0.0;
    for (double val : values)
        sq_sum += (val - mean) * (val - mean);
    return std::sqrt(sq_sum / values.size());
}

double hillClimbingFirst(double (*fitnessFunction)(const std::vector<double>&), double lower_bound, double upper_bound) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<double> perturbationDist(-STEP_SIZE, STEP_SIZE);
    std::uniform_int_distribution<int> indexDist(0, DIMENSION - 1);
    std::vector<double> currentSolution = generateRandomSolution(lower_bound, upper_bound, rng);
    double currentFitness = fitnessFunction(currentSolution);
    for (int iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {
        std::vector<double> neighbor = currentSolution;
        int randomIndex = indexDist(rng);
        double perturbation = perturbationDist(rng);
        neighbor[randomIndex] += perturbation;
        if (neighbor[randomIndex] < lower_bound) neighbor[randomIndex] = lower_bound;
        if (neighbor[randomIndex] > upper_bound) neighbor[randomIndex] = upper_bound;
        double neighborFitness = fitnessFunction(neighbor);
        if (neighborFitness < currentFitness) {
            currentSolution = neighbor;
            currentFitness = neighborFitness;
        }
    }
    return currentFitness;
}

double hillClimbingBest(double (*fitnessFunction)(const std::vector<double>&), double lower_bound, double upper_bound) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<double> perturbationDist(-STEP_SIZE, STEP_SIZE);
    std::vector<double> currentSolution = generateRandomSolution(lower_bound, upper_bound, rng);
    double currentFitness = fitnessFunction(currentSolution);
    for (int iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {
        double bestFitness = currentFitness;
        std::vector<double> bestNeighbor = currentSolution;
        for (int i = 0; i < DIMENSION; ++i) {
            std::vector<double> neighbor = currentSolution;
            double perturbation = perturbationDist(rng);
            neighbor[i] += perturbation;
            if (neighbor[i] < lower_bound) neighbor[i] = lower_bound;
            if (neighbor[i] > upper_bound) neighbor[i] = upper_bound;
            double neighborFitness = fitnessFunction(neighbor);
            if (neighborFitness < bestFitness) {
                bestFitness = neighborFitness;
                bestNeighbor = neighbor;
            }
        }
        currentSolution = bestNeighbor;
        currentFitness = bestFitness;
    }
    return currentFitness;
}

double hillClimbingWorst(double (*fitnessFunction)(const std::vector<double>&), double lower_bound, double upper_bound) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<double> perturbationDist(-STEP_SIZE, STEP_SIZE);
    std::vector<double> currentSolution = generateRandomSolution(lower_bound, upper_bound, rng);
    double currentFitness = fitnessFunction(currentSolution);
    for (int iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {
        double worstFitness = currentFitness;
        std::vector<double> worstNeighbor = currentSolution;
        for (int i = 0; i < DIMENSION; ++i) {
            std::vector<double> neighbor = currentSolution;
            double perturbation = perturbationDist(rng);
            neighbor[i] += perturbation;
            if (neighbor[i] < lower_bound) neighbor[i] = lower_bound;
            if (neighbor[i] > upper_bound) neighbor[i] = upper_bound;
            double neighborFitness = fitnessFunction(neighbor);
            if (neighborFitness > worstFitness) {
                worstFitness = neighborFitness;
                worstNeighbor = neighbor;
            }
        }
        currentSolution = worstNeighbor;
        currentFitness = worstFitness;
    }
    return currentFitness;
}

double simulatedAnnealing(double (*fitnessFunction)(const std::vector<double>&), double lower_bound, double upper_bound) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<double> perturbationDist(-STEP_SIZE, STEP_SIZE);
    std::uniform_int_distribution<int> indexDist(0, DIMENSION - 1);
    std::uniform_real_distribution<double> probabilityDist(0.0, 1.0);
    std::vector<double> currentSolution = generateRandomSolution(lower_bound, upper_bound, rng);
    double currentFitness = fitnessFunction(currentSolution);
    double temperature = 100.0;
    double coolingRate = 0.995;
    for (int iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {
        std::vector<double> neighbor = currentSolution;
        int randomIndex = indexDist(rng);
        double perturbation = perturbationDist(rng);
        neighbor[randomIndex] += perturbation;
        if (neighbor[randomIndex] < lower_bound) neighbor[randomIndex] = lower_bound;
        if (neighbor[randomIndex] > upper_bound) neighbor[randomIndex] = upper_bound;
        double neighborFitness = fitnessFunction(neighbor);
        if (neighborFitness < currentFitness || probabilityDist(rng) < std::exp((currentFitness - neighborFitness) / temperature)) {
            currentSolution = neighbor;
            currentFitness = neighborFitness;
        }
        temperature *= coolingRate;
        if (temperature < 1e-5) break;
    }
    return currentFitness;
}

int main() 
{
    std::cout << std::fixed << std::setprecision(6);
    std::random_device rd;
    std::mt19937 rng(rd());
    std::cout << DIMENSION << "-dimensional\n";
    std::vector<double> firstResults, bestResults, worstResults, saResults;

    for (int i = 0; i < RUNS; ++i) {
        firstResults.push_back(hillClimbingFirst(schwefel, SCHWEFEL_LOWER_BOUND, SCHWEFEL_UPPER_BOUND));
        bestResults.push_back(hillClimbingBest(schwefel, SCHWEFEL_LOWER_BOUND, SCHWEFEL_UPPER_BOUND));
        worstResults.push_back(hillClimbingWorst(schwefel, SCHWEFEL_LOWER_BOUND, SCHWEFEL_UPPER_BOUND));
        saResults.push_back(simulatedAnnealing(schwefel, SCHWEFEL_LOWER_BOUND, SCHWEFEL_UPPER_BOUND));
    }

    auto worstMin = *std::min_element(firstResults.begin(), firstResults.end());
    auto worstMax = *std::max_element(firstResults.begin(), firstResults.end());
    double worstMean = calculateMean(firstResults);
    double worstStdDev = calculateStandardDeviation(firstResults, worstMean);

    std::cout << "Mean: " << worstMean << ", StdDev: " << worstStdDev << ", Min: " << worstMin << ", Max: " << worstMax << "\n";
    return 0;
}

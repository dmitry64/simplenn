#include "simpleneuralnet.hpp"

#include <chrono>

using InputValues = NeuralNet::InputValues;
using OutputValues = NeuralNet::OutputValues;
using IO = NeuralNet::IO;

int main()
{
    std::cout << "Starting neural network simulation...\n";

    // Neural net topology:
    // O*O*O\
    // O*O*O-O
    // O*O*O/
    // O*O*O/
    // =======
    // 4 4 4 1
    NeuralNetParams params(0.15,0.5);
    NeuralNet nn({4, 4, 4, 1},params);

    // Training set
    std::vector<IO> data;

    // A && B && C && D
    data.push_back(IO(InputValues({{0,0,0,0}}),OutputValues({0})));
    data.push_back(IO(InputValues({{0,0,1,0}}),OutputValues({0})));
    data.push_back(IO(InputValues({{0,1,0,0}}),OutputValues({0})));
    data.push_back(IO(InputValues({{0,1,1,0}}),OutputValues({0})));
    data.push_back(IO(InputValues({{1,0,0,0}}),OutputValues({0})));
    data.push_back(IO(InputValues({{1,0,1,0}}),OutputValues({0})));
    data.push_back(IO(InputValues({{1,1,0,0}}),OutputValues({0})));
    data.push_back(IO(InputValues({{1,1,1,0}}),OutputValues({0})));
    data.push_back(IO(InputValues({{0,0,0,1}}),OutputValues({0})));
    data.push_back(IO(InputValues({{0,0,1,1}}),OutputValues({0})));
    data.push_back(IO(InputValues({{0,1,0,1}}),OutputValues({0})));
    data.push_back(IO(InputValues({{0,1,1,1}}),OutputValues({0})));
    data.push_back(IO(InputValues({{1,0,0,1}}),OutputValues({0})));
    data.push_back(IO(InputValues({{1,0,1,1}}),OutputValues({0})));
    data.push_back(IO(InputValues({{1,1,0,1}}),OutputValues({0})));
    data.push_back(IO(InputValues({{1,1,1,1}}),OutputValues({1})));

    auto start = std::chrono::system_clock::now();

    // Learning...
    for(unsigned int i = 0; i<1000000; ++i) {
        for(auto & pair : data) {
            nn.feedForward(pair.first);
            nn.backProp(pair.second);
        }
    }

    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Learning done! time: " << elapsed.count() << " ms\n";

    nn.printDebug();

    // Test our neural net
    InputValues inputVals { {1.0, 1.0, 1.0, 1.0} }; // 1 && 1 && 1 && 1 = 1
    nn.feedForward(inputVals);

    OutputValues resultVals;
    nn.getResults(resultVals);

    for(auto val : resultVals) {
        std::cout << "Result: " << std::setprecision (3) <<  std::fixed  << (val) << "\n";
    }

    std::cout << "Simulation finished!" << std::endl;

    return 0;
}

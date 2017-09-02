#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <chrono>

const static double eta = 0.15;
const static double alpha = 0.5;

const static unsigned int InputNodesCount = 3;
const static unsigned int OutputNodesCount = 1;

typedef std::array<double, InputNodesCount> InputValues;
typedef std::array<double, OutputNodesCount> OutputValues;
typedef std::pair<InputValues, OutputValues> IO;
typedef std::vector<unsigned int> Topology;

class Neuron;
typedef std::vector<Neuron> Layer;

class Neuron{
    struct Connection {
        double weight;
        double deltaWeight;
    };
public:
    Neuron(unsigned int numOutputs, unsigned int myIndex);
    void setOutputVal(double val) { _outValue = val; }
    double getOutputVal() const { return _outValue; }
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer) const;
    double getWeightedOutputForIndex(unsigned int index) const;
private:
    static double transferFunction(double x);
    static double transferDerivative(double x);
    static double randomWeight();
    double sumDOW(const Layer & nextLayer);
    unsigned int _myIndex;
    double _outValue;
    double _gradient;
    std::vector<Connection> _outputWeights;
};

Neuron::Neuron(unsigned int numOutputs, unsigned int myIndex)
{
    _outValue = 0;
    _gradient = 0;
    _myIndex = myIndex;

    for(unsigned int c = 0; c < numOutputs; ++c) {
        Connection connection;
        connection.weight = randomWeight();
        connection.deltaWeight = 0;
        _outputWeights.push_back(connection);
    }
}

void Neuron::feedForward(const Layer &prevLayer)
{
    double sum = 0.0;
    for(auto & neuron : prevLayer) {
        sum += neuron.getWeightedOutputForIndex(_myIndex);
    }
    _outValue = Neuron::transferFunction(sum);
}

void Neuron::calcOutputGradients(double targetVal)
{
    _gradient = (targetVal - _outValue) * Neuron::transferDerivative(_outValue);
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    _gradient = sumDOW(nextLayer) * Neuron::transferDerivative(_outValue);
}

void Neuron::updateInputWeights(Layer &prevLayer) const
{
    for(auto & neuron : prevLayer) {
        Connection & curConnection = neuron._outputWeights[_myIndex];
        double newDeltaWeight = eta * neuron.getOutputVal() * _gradient + alpha * curConnection.deltaWeight;
        curConnection.deltaWeight = newDeltaWeight;
        curConnection.weight += newDeltaWeight;
    }
}

double Neuron::getWeightedOutputForIndex(unsigned int index) const
{
    return _outValue * _outputWeights[index].weight;
}

double Neuron::transferFunction(double x)
{
    return tanh(x);
}

double Neuron::transferDerivative(double x)
{
    return 1.0 - x * x;
}

double Neuron::randomWeight()
{
    return double(rand()) / double(RAND_MAX);
}

double Neuron::sumDOW(const Layer &nextLayer)
{
    double sum = 0.0;
    for(unsigned int n = 0; n < nextLayer.size() - 1; ++n) {
        sum += _outputWeights[n].weight * nextLayer[n]._gradient;
    }
    return sum;
}

class NeuralNet{
public:
    NeuralNet(const Topology & topology);
    void feedForward(const InputValues & inputVals);
    void backProp(const OutputValues &targetVals);
    void getResults(OutputValues &resultVals) const;
    void printDebug() const;
private:
    std::vector<Layer> _layers;
    double _error;
    double _recentAverageSmoothingFactor;
    double _recentAverageError;
};

NeuralNet::NeuralNet(const Topology &topology)
{
    unsigned int numLayers = static_cast<unsigned int>(topology.size());
    for(unsigned int layerNum = 0; layerNum < numLayers; ++layerNum) {
        Layer layer;
        unsigned int numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
        for(unsigned int num = 0; num <= topology[layerNum]; ++num) {
            layer.push_back(Neuron(numOutputs, num));
        }
        layer.back().setOutputVal(1.0);
        _layers.push_back(layer);
    }
}

void NeuralNet::feedForward(const InputValues &inputVals)
{
    unsigned int size = static_cast<unsigned int>(inputVals.size());
    assert(size == _layers[0].size() - 1);

    for(unsigned int i = 0; i < size; ++i) {
        _layers[0][i].setOutputVal(inputVals[i]);
    }

    auto it1 = _layers.begin();
    auto it2 = _layers.begin() + 1;
    while( it2!=_layers.end() ) {
        Layer & prevLayer = it1.operator*();
        Layer & layer = it2.operator*();
        for(auto it=layer.begin(); it!=layer.end() - 1; ++it) {
            it.operator*().feedForward(prevLayer);
        }
        ++it2;
        ++it1;
    }
}

void NeuralNet::backProp(const OutputValues &targetVals)
{
    assert(_layers.size() > 1);
    Layer & outputLayer = _layers.back();
    _error = 0.0;
    for(unsigned int n = 0; n < outputLayer.size() - 1; ++n) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        _error += delta * delta;
    }
    _error /= outputLayer.size() - 1;
    _error = sqrt(_error);
    _recentAverageError = (_recentAverageError * _recentAverageSmoothingFactor + _error) / (_recentAverageSmoothingFactor + 1.0);

    for(unsigned int n = 0; n < outputLayer.size() - 1; ++n) {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    for(unsigned int layerNum = _layers.size() - 2; layerNum > 0; --layerNum) {
        Layer & hiddenLayer = _layers[layerNum];
        const Layer & nextLayer = _layers[layerNum + 1];

        for(auto & hiddenNeuron : hiddenLayer) {
            hiddenNeuron.calcHiddenGradients(nextLayer);
        }
    }

    for(auto it2 = _layers.rbegin(); it2 != _layers.rend() - 1; ++it2) {
        const Layer & layer = it2.operator*();
        Layer & prevLayer = (it2 + 1).operator*();

        for(auto it = layer.begin(); it != layer.end() - 1; ++it) {
            it.operator*().updateInputWeights(prevLayer);
        }
    }
}

void NeuralNet::getResults(OutputValues &resultVals) const
{
    for(unsigned int n = 0; n < _layers.back().size() - 1; ++n) {
        resultVals.at(n) = (_layers.back()[n].getOutputVal());
    }
}

void NeuralNet::printDebug() const
{
    std::cout << std::setprecision(8) << std::fixed  << "Learning error: " << _error
              << " average: " << _recentAverageError << '\n';
}

int main()
{
    std::cout << "Starting neural network simulation...\n";

    // Neural net topology:
    // O*O\
    // O*O-O
    // O*O/
    Topology topology {InputNodesCount, 3, OutputNodesCount};

    NeuralNet nn(topology);

    // Training set
    std::vector<IO> data;

    // A && B && C
    data.push_back(IO(InputValues({{0,0,0}}),OutputValues({{0}})));
    data.push_back(IO(InputValues({{0,0,1}}),OutputValues({{0}})));
    data.push_back(IO(InputValues({{0,1,0}}),OutputValues({{0}})));
    data.push_back(IO(InputValues({{0,1,1}}),OutputValues({{0}})));
    data.push_back(IO(InputValues({{1,0,0}}),OutputValues({{0}})));
    data.push_back(IO(InputValues({{1,0,1}}),OutputValues({{0}})));
    data.push_back(IO(InputValues({{1,1,0}}),OutputValues({{0}})));
    data.push_back(IO(InputValues({{1,1,1}}),OutputValues({{1}})));

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
    InputValues inputVals { {1.0, 1.0, 1.0} }; // 1 && 1 && 1 = 1
    nn.feedForward(inputVals);

    OutputValues resultVals;
    nn.getResults(resultVals);

    for(auto val : resultVals) {
        std::cout << "Result: " << std::setprecision (3) <<  std::fixed  << (val) << "\n";
    }

    std::cout << "Simulation finished!" << std::endl;

    return 0;
}

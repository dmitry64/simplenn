#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <chrono>

static double eta = 0.15;
static double alpha = 0.5;

struct Connection {
    double weight;
    double deltaWeight;
};

class Neuron;

typedef std::vector<Neuron> Layer;
typedef std::vector<double> Values;
typedef std::pair<Values,Values> IO;

class Neuron{
public:
    Neuron(unsigned int numOutputs, unsigned int myIndex);
    void setOutputVal(double val) { _outValue = val; }
    double getOutputVal() const { return _outValue; }
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer) const;
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

    for(auto & neuron : prevLayer ) {
        sum += neuron.getOutputVal() * neuron._outputWeights[_myIndex].weight;
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
    for(unsigned int n =0; n < nextLayer.size() - 1; ++n) {
        sum += _outputWeights[n].weight * nextLayer[n]._gradient;
    }

    return sum;
}

class NeuralNet{
public:
    NeuralNet(const std::vector<unsigned int> & topology);
    void feedForward(const Values & inputVals);
    void backProp(const Values & targetVals);
    void getResults(Values &resultVals) const;
    void printDebug() const;
private:
    std::vector<Layer> _layers;
    double _error;
    double _recentAverageSmoothingFactor;
    double _recentAverageError;
};

NeuralNet::NeuralNet(const std::vector<unsigned int> &topology)
{
    unsigned int numLayers = topology.size();
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

void NeuralNet::feedForward(const Values &inputVals)
{
    unsigned int size = inputVals.size();
    assert(size == _layers[0].size() - 1);

    for(unsigned int i = 0; i < size; ++i) {
        _layers[0][i].setOutputVal(inputVals[i]);
    }

    for(unsigned int layerNum = 1; layerNum < _layers.size(); ++layerNum) {
        Layer & layer = _layers[layerNum];
        Layer & prevLayer = _layers[layerNum - 1];

        for(unsigned int n = 0; n < layer.size() - 1; ++n) {
            layer[n].feedForward(prevLayer);
        }
    }
}

void NeuralNet::backProp(const Values &targetVals)
{
    Layer & outputLayer = _layers.back();
    const unsigned int size = outputLayer.size() - 1;
    _error = 0.0;
    for(unsigned n = 0; n < size; ++n) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        _error += delta * delta;
    }
    _error /= size;
    _error = sqrt(_error);

    _recentAverageError = (_recentAverageError * _recentAverageSmoothingFactor + _error) / (_recentAverageSmoothingFactor + 1.0);

    for(unsigned int n = 0; n < size; ++n) {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    for(unsigned int layerNum = _layers.size() - 2; layerNum > 0; --layerNum) {
        Layer & hiddenLayer = _layers[layerNum];
        const Layer & nextLayer = _layers[layerNum + 1];

        for(auto & hiddenNeuron : hiddenLayer) {
            hiddenNeuron.calcHiddenGradients(nextLayer);
        }
    }

    for(unsigned int layerNum = _layers.size() - 1; layerNum > 0; --layerNum) {
        const Layer & layer = _layers[layerNum];
        Layer & prevLayer = _layers[layerNum - 1];

        for(unsigned int n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void NeuralNet::getResults(Values &resultVals) const
{
    resultVals.clear();
    for(unsigned int n = 0; n < _layers.back().size() - 1; ++n) {
        resultVals.push_back(_layers.back()[n].getOutputVal());
    }
}

void NeuralNet::printDebug() const
{
    std::cout << std::setprecision(8) << std::fixed  << "error: " << _error
              << " average: " << _recentAverageError << '\n';
}

int main()
{
    std::cout << "Starting neural network simulation...\n";

    std::vector<unsigned int> topology {2,3,1};
    NeuralNet nn(topology);

    std::vector<IO> data;

    data.push_back(IO(Values({0,0}),Values({0})));
    data.push_back(IO(Values({1,1}),Values({1})));
    data.push_back(IO(Values({1,0}),Values({0})));
    data.push_back(IO(Values({0,1}),Values({0})));
    data.push_back(IO(Values({0,0}),Values({0})));
    data.push_back(IO(Values({1,0}),Values({0})));
    data.push_back(IO(Values({1,1}),Values({1})));
    data.push_back(IO(Values({0,1}),Values({0})));

    auto start = std::chrono::system_clock::now();

    for(int i = 0; i<1000000; ++i) {
        for(auto & pair : data) {
            nn.feedForward(pair.first);
            nn.backProp(pair.second);
        }
    }

    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << elapsed.count() << " ms\n";

    nn.printDebug();

    Values inputVals { 1.0, 1.0 };
    nn.feedForward(inputVals);

    Values resultVals;
    nn.getResults(resultVals);

    for(auto val : resultVals) {
        std::cout << "result: " << std::setprecision (3) <<  std::fixed  << (val) << "\n";
    }

    return 0;
}

#ifndef SIMPLENEURALNET_HPP
#define SIMPLENEURALNET_HPP

#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <cstdlib>
#include <cassert>
#include <cmath>

using FloatingPointType = double;

struct NeuralNetParams {
    const FloatingPointType _eta;
    const FloatingPointType _alpha;
    NeuralNetParams (FloatingPointType eta, FloatingPointType alpha) : _eta(eta), _alpha(alpha) { }
};

class Neuron {
    const NeuralNetParams & _params;
    struct Connection {
        FloatingPointType weight;
        FloatingPointType deltaWeight;
    };
public:
    using Layer = std::vector<Neuron>;

    Neuron(unsigned int numOutputs, unsigned int myIndex, const NeuralNetParams & params) : _params(params)
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
    void setOutputVal(FloatingPointType val) { _outValue = val; }
    FloatingPointType getOutputVal() const { return _outValue; }
    void feedForward(const Layer &prevLayer)
    {
        FloatingPointType sum = 0.0;
        for(auto & neuron : prevLayer) {
            sum += neuron.getWeightedOutputForIndex(_myIndex);
        }
        _outValue = Neuron::transferFunction(sum);
    }
    void calcOutputGradients(FloatingPointType targetVal)
    {
        _gradient = (targetVal - _outValue) * Neuron::transferDerivative(_outValue);
    }
    void calcHiddenGradients(const Layer &nextLayer)
    {
        _gradient = sumDOW(nextLayer) * Neuron::transferDerivative(_outValue);
    }
    void updateInputWeights(Layer &prevLayer) const
    {
        for(auto & neuron : prevLayer) {
            Connection & curConnection = neuron._outputWeights[_myIndex];
            FloatingPointType newDeltaWeight = _params._eta * neuron.getOutputVal() * _gradient + _params._alpha * curConnection.deltaWeight;
            curConnection.deltaWeight = newDeltaWeight;
            curConnection.weight += newDeltaWeight;
        }
    }
    FloatingPointType getWeightedOutputForIndex(unsigned int index) const
    {
        return _outValue * _outputWeights[index].weight;
    }
private:
    static FloatingPointType transferFunction(FloatingPointType x)
    {
        return tanh(x);
    }
    static FloatingPointType transferDerivative(FloatingPointType x)
    {
        return 1.0 - x * x;
    }
    static FloatingPointType randomWeight()
    {
        return FloatingPointType(rand()) / FloatingPointType(RAND_MAX);
    }
    FloatingPointType sumDOW(const Layer & nextLayer)
    {
        FloatingPointType sum = 0.0;
        for(unsigned int n = 0; n < nextLayer.size() - 1; ++n) {
            sum += _outputWeights[n].weight * nextLayer[n]._gradient;
        }
        return sum;
    }
    unsigned int _myIndex;
    FloatingPointType _outValue;
    FloatingPointType _gradient;
    std::vector<Connection> _outputWeights;
};

class NeuralNet {
public:
    using InputValues = std::vector<FloatingPointType>;
    using OutputValues = std::vector<FloatingPointType>;
    using NeuronType = Neuron;
    using Layer = std::vector<Neuron>;
    using Layers = std::vector<Layer>;
    using IO = std::pair<InputValues, OutputValues>;
    using Topology = std::vector<unsigned int>;

    NeuralNet(const Topology & topology, NeuralNetParams & params) : _params(params)
    {
        assert(topology.size());
        unsigned int numLayers = static_cast<unsigned int>(topology.size());
        for(unsigned int layerNum = 0; layerNum < numLayers; ++layerNum) {
            Layer layer;
            unsigned int numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
            for(unsigned int num = 0; num <= topology[layerNum]; ++num) {
                layer.push_back(NeuronType(numOutputs, num, _params));
            }
            layer.back().setOutputVal(1.0);
            _layers.push_back(layer);
        }
    }
    void feedForward(const InputValues & inputVals)
    {
        unsigned int size = static_cast<unsigned int>(inputVals.size());
        assert(size == _layers.front().size() - 1);

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
    void backProp(const OutputValues &targetVals)
    {
        assert(targetVals.size() == _layers.back().size() - 1);
        Layer & outputLayer = _layers.back();
        _error = 0.0;
        for(unsigned int n = 0; n < outputLayer.size() - 1; ++n) {
            FloatingPointType delta = targetVals[n] - outputLayer[n].getOutputVal();
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
    void getResults(OutputValues &resultVals) const
    {
        resultVals.clear();
        for(unsigned int n = 0; n < _layers.back().size() - 1; ++n) {
            resultVals.push_back(_layers.back()[n].getOutputVal());
        }
    }
    void printDebug() const
    {
        std::cout << std::setprecision(8) << std::fixed  << "Learning error: " << _error
                  << " average: " << _recentAverageError << '\n';
    }
private:
    const NeuralNetParams _params;
    Layers _layers;
    FloatingPointType _error;
    FloatingPointType _recentAverageSmoothingFactor;
    FloatingPointType _recentAverageError;
};

#endif // SIMPLENEURALNET_HPP

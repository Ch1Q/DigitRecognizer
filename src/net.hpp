#pragma once
#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <functional>
#include <memory>
struct Sample
{
    std::vector<double> features;
    std::vector<double> labels;
};

class Nueron
{
    double output;
    double delta;

public:
};

class Layer
{
    std::vector<Nueron> nuerons;
    std::vector<size_t> shape;
    std::vector<size_t> strides;

    void computeStrides(std::vector<size_t> shp);

public:
    Layer(std::vector<size_t> shp);
};

class Synapse
{
    double weight;
    double bias;
    double gradient;
    size_t fromIdx;
    size_t toIdx;
};

class Link
{
    Layer &sourceLayer;
    Layer &targetLayer;
    std::vector<Synapse> synapses;
    std::function<double(double)> activate;
    virtual void initSynapses() = 0;

public:
    Link(Layer &src, Layer &tgt, std::function<double(double)> atv = [](double v)
                                 { return v; });
};

class Network
{
    std::unordered_map<std::string, std::shared_ptr<Layer>> layers;
    std::vector<std::shared_ptr<Link>> links;
};
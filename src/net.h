#pragma once

#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <functional>
#include <memory>
#include <immintrin.h>
#include <optional>

#pragma pack(push, 1)
struct FileHeader
{
    uint32_t magic;      // 魔数：0xDEADBEEF（自定义）
    uint16_t version;    // 版本号：1
    uint32_t num_layers; // 层数量
    uint32_t num_links;  // 链接数量
};
#pragma pack(pop) // 恢复默认对齐

struct Sample
{
    std::vector<double> features;
    std::vector<double> labels;
};

class SampleSet
{

    std::vector<Sample> samples;
    friend Network;

public:
    SampleSet(size_t featureSize_, size_t labelSize_);
    SampleSet(size_t featureSize_, size_t labelSize_, size_t sampleSize);
    void resize(size_t size_);
    size_t size();
    bool push_back(Sample sample);
    void clear();
    void reserve(size_t size);
    const Sample at(size_t);
    const size_t featureSize;
    const size_t labelSize;
    std::vector<Sample> getSamples();
};

class Network
{

    class Layer;

    class Link;

    class DenseLink;

    std::vector<Layer> m_layers;
    std::vector<Link> m_links;
    std::vector<std::vector<size_t>> m_forwardCache;
    std::vector<std::vector<size_t>> m_backwardCache;
    void updateForwardCache();
    void clearForwardCache();
    void updateBackwardCache();
    void clearBackwardCache();
    friend Link;

public:
    std::string comment;
    Network(std::shared_ptr<Layer> input, std::shared_ptr<Layer> output);
    Network(std::string path);
    // void saveModel(const std::string &file);
    Sample predict(const Sample &sample);
    bool train(const SampleSet &sampleSet);
    void printLayersInfo();
    void printLinksInfo();
};

static double sigmoid(double x);

static double sigmoidDeri(double x);

static double linear(double x);

static double linearDeri(double x);

const std::unordered_map<std::string, std::function<double(double)>> activateFunc =
    {
        {"sigmoid", sigmoid},
        {"linear", linear}};

const std::unordered_map<std::string, std::function<double(double)>> activateDeriFunc =
    {
        {"sigmoidDeri", sigmoidDeri},
        {"linearDeri", linearDeri}};

class Network::Layer
{
    struct Nueron
    {
        double delta = 0;
        double input = 0;
        double output = 0;
        double bias = 0;
    };

    std::vector<Nueron> m_neurons;
    std::vector<size_t> m_shape;
    std::vector<size_t> m_strides;
    std::string m_activate;
    void initStrides();
    friend Network::Link;

public:
    Layer(std::vector<size_t> shape, std::string activate = "linear");
    const size_t &size();
    const std::vector<size_t> &shape();
    std::string comment;
};

class Network::Link
{
    struct Synapse
    {
        double weight;
        double gradient;
        size_t fromIdx;
        size_t toIdx;
    };

protected:
    size_t m_source;
    size_t m_target;
    std::vector<Synapse> m_synapses;
    virtual void initSynapses();

public:
    Link(size_t source, size_t target);
    const size_t &source();
    const size_t &target();
    virtual ~Link();
};

class Network::DenseLink : public Link
{

    void initSynapses() override;

public:
    DenseLink(size_t source, size_t target);
    void normalInitSynapses();
    void valueInitSynapses(double value);
};

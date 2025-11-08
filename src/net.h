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

double sigmoid(double x);

double sigmoidDeri(double x);

double linear(double x);

double linearDeri(double x);

const std::unordered_map<std::string, std::function<double(double)>> activateFunc =
    {
        {"sigmoid", sigmoid},
        {"linear", linear}};

const std::unordered_map<std::string, std::function<double(double)>> activateDeriFunc =
    {
        {"sigmoidDeri", sigmoidDeri},
        {"linearDeri", linearDeri}};

struct Sample
{
    std::vector<double> features;
    std::vector<double> labels;
};

class SampleSet
{

    std::vector<Sample> samples;

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
};

class Nueron
{

public:
    double value = 0;
    double delta = 0;
};

class Layer
{
    std::vector<Nueron> neurons;
    std::vector<size_t> shape;
    std::vector<size_t> strides;

    void initStrides(std::vector<size_t> shp);
    friend Network;

public:
    std::string comment;
    Layer(std::vector<size_t> shp);
    size_t size();
    std::vector<size_t> getShape();
};

class Synapse
{
public:
    double weight;
    double bias;
    double gradient;
    size_t fromIdx;
    size_t toIdx;
};

class Link
{
protected:
    std::shared_ptr<Layer> sourceLayer;
    std::shared_ptr<Layer> targetLayer;
    std::vector<Synapse> m_synapses;
    std::string m_activate;
    virtual void initSynapses();

public:
    Link(std::shared_ptr<Layer> src, std::shared_ptr<Layer> tgt, std::string acti = "linear");
    std::shared_ptr<Layer> source();
    std::shared_ptr<Layer> target();
    const std::vector<Synapse> synapses();
    const std::string activate();
    virtual ~Link();
};

class DenseLink : public Link
{

    void initSynapses() override;

public:
    DenseLink(std::shared_ptr<Layer> src, std::shared_ptr<Layer> tgt, std::string acti = "linear");
    void normalInitSynapses();
};

class Network
{
    std::vector<std::shared_ptr<Layer>> layers;
    std::vector<std::shared_ptr<Link>> links;

public:
    std::string comment;
    Network(std::shared_ptr<Layer> input, std::shared_ptr<Layer> output);
    void addLayer(std::shared_ptr<Layer> layer);
    void addLink(std::shared_ptr<Link> link);
    void saveModel(const std::string &file);
    // void loadModel(const std::string &file);
    Sample predict(const Sample &sample);
};

void normalInitSynapses(std::vector<Synapse> &syns);

bool loadSamples(std::string path, SampleSet &samples);

void printSampleSet(SampleSet sampleSet);

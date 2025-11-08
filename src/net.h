#pragma once
#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <functional>
#include <memory>
#include <immintrin.h>
#include <optional>

template <typename KeyType, typename ValueType>
class BiDirectionalMap
{
private:
    // 正向映射：Key → Value
    std::unordered_map<KeyType, ValueType> key_to_value_;
    // 反向映射：Value → Key
    std::unordered_map<ValueType, KeyType> value_to_key_;

public:
    // --------------------------
    // 插入元素（键/值均需唯一）
    // --------------------------
    bool insert(const KeyType &key, const ValueType &value)
    {
        // 检查键或值是否已存在
        if (key_to_value_.contains(key) || value_to_key_.contains(value))
        {
            return false; // 插入失败（重复）
        }
        // 同步插入双向映射
        key_to_value_[key] = value;
        value_to_key_[value] = key;
        return true; // 插入成功
    }

    // --------------------------
    // 删除元素（支持按Key或Value删除）
    // --------------------------
    void erase_by_key(const KeyType &key)
    {
        if (auto it = key_to_value_.find(key); it != key_to_value_.end())
        {
            value_to_key_.erase(it->second); // 同步删除反向映射
            key_to_value_.erase(it);         // 删除正向映射
        }
    }

    void erase_by_value(const ValueType &value)
    {
        if (auto it = value_to_key_.find(value); it != value_to_key_.end())
        {
            key_to_value_.erase(it->second); // 同步删除正向映射
            value_to_key_.erase(it);         // 删除反向映射
        }
    }

    // --------------------------
    // 查找元素（安全返回std::optional）
    // --------------------------
    std::optional<ValueType> find_value(const KeyType &key) const
    {
        if (auto it = key_to_value_.find(key); it != key_to_value_.end())
        {
            return it->second;
        }
        return std::nullopt; // 未找到
    }

    std::optional<KeyType> find_key(const ValueType &value) const
    {
        if (auto it = value_to_key_.find(value); it != value_to_key_.end())
        {
            return it->second;
        }
        return std::nullopt; // 未找到
    }

    // --------------------------
    // 其他常用接口
    // --------------------------
    size_t size() const noexcept { return key_to_value_.size(); }
    bool empty() const noexcept { return key_to_value_.empty(); }
    void clear() noexcept
    {
        key_to_value_.clear();
        value_to_key_.clear();
    }

    // --------------------------
    // 遍历所有键值对（与std::unordered_map一致）
    // --------------------------
    auto begin() const noexcept { return key_to_value_.begin(); }
    auto end() const noexcept { return key_to_value_.end(); }
};

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
    double output = 0;
    double delta = 0;
};

class Layer
{
    std::vector<Nueron> neurons;
    std::vector<size_t> shape;
    std::vector<size_t> strides;

    void initStrides(std::vector<size_t> shp);

public:
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

    BiDirectionalMap<std::string, std::shared_ptr<Layer>> layers;
    std::vector<std::shared_ptr<Link>> links;

public:
    Network(std::shared_ptr<Layer> input, std::shared_ptr<Layer> output);
    bool addLayer(std::string name, std::shared_ptr<Layer> lyr);
    bool addLink(std::shared_ptr<Link> lnk);
    bool saveModel(const std::string &file);
    // void loadModel(const std::string &file);
    bool predict(const Sample &smp);
};

void normalInitSynapses(std::vector<Synapse> &syns);

bool loadSamples(std::string path, SampleSet &samples);

void printSampleSet(SampleSet sampleSet);

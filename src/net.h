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
public:
    std::shared_ptr<Layer> sourceLayer;
    std::shared_ptr<Layer> targetLayer;
    std::vector<Synapse> synapses;
    std::function<double(double)> activate;
    virtual void initSynapses();

    Link(std::shared_ptr<Layer> src, std::shared_ptr<Layer> tgt, std::function<double(double)> atv = [](double v)
                                                                 { return v; });
    std::shared_ptr<Layer> source();
    std::shared_ptr<Layer> target();
    virtual ~Link();
};

class DenseLink : public Link
{

    void initSynapses() override;

public:
    DenseLink(std::shared_ptr<Layer> src, std::shared_ptr<Layer> tgt, std::function<double(double)> atv = [](double v)
                                                                      { return v; });
    void normalInitSynapses();
};

class Network
{
public:
    BiDirectionalMap<std::string, std::shared_ptr<Layer>> layers;
    std::vector<std::shared_ptr<Link>> links;

    Network(std::shared_ptr<Layer> input, std::shared_ptr<Layer> output);
    void addLayer(std::string name, std::shared_ptr<Layer> lyr);
    void addLink(std::shared_ptr<Link> lnk);
    void saveModel(const std::string &file);
    void loadModel(const std::string &file);
};

void normalInitSynapses(std::vector<Synapse> &syns);

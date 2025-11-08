#include "net.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <numeric>

SampleSet::SampleSet(size_t featureSize_, size_t labelSize_) : featureSize(featureSize_), labelSize(labelSize_)
{
}

SampleSet::SampleSet(size_t featureSize_, size_t labelSize_, size_t sampleSize) : featureSize(featureSize_), labelSize(labelSize_)
{
    samples.resize(sampleSize);
}

void SampleSet::resize(size_t size)
{
    samples.resize(size);
}

size_t SampleSet::size()
{
    return samples.size();
}

bool SampleSet::push_back(Sample sample_)
{
    if (sample_.features.size() == featureSize && sample_.labels.size() == labelSize)
    {
        samples.push_back(sample_);
        return 1;
    }
    else
    {
        std::cout << "Error: sample's Size didn't match the set" << std::endl;
        return 0;
    }
}

void SampleSet::clear()
{
    samples.clear();
}

void SampleSet::reserve(size_t size)
{
    samples.reserve(size);
}

const Sample SampleSet::at(size_t size)
{
    if (size >= samples.size())
    {
        std::cout << "Sample.at Error: " << "out of limit size: " << samples.size();
        return Sample();
    }
    return samples.at(size);
}

void Layer::initStrides(std::vector<size_t> shp)
{
    size_t sz = shape.size();
    strides.resize(sz);
    strides.back() = 1;
    for (int i = 1; i < sz; i++)
    {
        strides[sz - i - 1] = strides[sz - i] * shape[sz - i];
    }
}

Layer::Layer(std::vector<size_t> shp)
{
    shape = shp;
    initStrides(shp);
    size_t nr_sz = std::accumulate(shp.begin(), shp.end(), 1ULL, [](size_t a, size_t b)
                                   { return a * b; });
    neurons.resize(nr_sz);
}

size_t Layer::size()
{
    return neurons.size();
}

std::vector<size_t> Layer::getShape()
{
    return shape;
}

Link::Link(std::shared_ptr<Layer> src, std::shared_ptr<Layer> tgt, std::string acti) : sourceLayer(src), targetLayer(tgt), m_activate(acti)
{
}

Link::~Link()
{
}

void Link::initSynapses()
{
}

std::shared_ptr<Layer> Link::source()
{
    return sourceLayer;
}

std::shared_ptr<Layer> Link::target()
{
    return targetLayer;
}

const std::vector<Synapse> Link::synapses()
{
    return m_synapses;
}
const std::string Link::activate()
{
    return m_activate;
}
DenseLink::DenseLink(std::shared_ptr<Layer> src, std::shared_ptr<Layer> tgt, std::string acti) : Link(src, tgt, acti)
{
    initSynapses();
}

void DenseLink::initSynapses()
{
    size_t sy_sz = sourceLayer->size() * targetLayer->size();
    m_synapses.resize(sy_sz);
    for (size_t j = 0; j < targetLayer->size(); j++)
    {
        for (size_t i = 0; i < sourceLayer->size(); i++)
        {
            m_synapses.at(j + i * targetLayer->size()).fromIdx = j;
            m_synapses.at(j + i * targetLayer->size()).toIdx = i;
        }
    }
}

void DenseLink::normalInitSynapses()
{
    ::normalInitSynapses(m_synapses);
}

Network::Network(std::shared_ptr<Layer> input, std::shared_ptr<Layer> output)
{
    addLayer("inputLayer", input);
    addLayer("outputLayer", output);
}

void Network::addLayer(std::string name, std::shared_ptr<Layer> lyr)
{
    layers.insert(name, lyr);
}

void Network::addLink(std::shared_ptr<Link> lnk)
{
    links.push_back(lnk);
}

bool Network::predict(const Sample &smp)
{
    if (smp.features.size() != layers.find_value("input"))
}

//
//
//
//
//
//
//
void Network::saveModel(const std::string &filename)
{
    // 1. 处理文件名：确保以.nll结尾
    std::string final_filename = filename;
    if (final_filename.find(".nll") == std::string::npos)
    {
        final_filename += ".nll"; // 自动添加.nll后缀
    }

    // 2. 打开文件（二进制模式 + 截断）
    std::ofstream ofs(final_filename, std::ios::binary | std::ios::trunc);
    if (!ofs)
    {
        throw std::runtime_error("无法打开文件写入: " + final_filename);
    }

    // 3. 写入文件头（魔数+版本+数量）
    FileHeader header{};
    header.magic = 0x20041022; // 自定义魔数（标识.nll文件）
    header.version = 1;        // 版本号
    header.num_layers = layers.size();
    header.num_links = links.size();

    // 写入文件头（转为char*，避免对齐问题）
    ofs.write(reinterpret_cast<const char *>(&header), sizeof(header));

    // 4. 写入所有层信息（名称+形状）
    for (const auto &[name, layer] : layers)
    {
        // 写入层名称（长度+内容）
        uint32_t name_len = static_cast<uint32_t>(name.size());
        ofs.write(reinterpret_cast<const char *>(&name_len), sizeof(name_len));
        ofs.write(name.c_str(), name_len);

        // 写入层形状（长度+维度数组）
        const auto &shape = layer->getShape();
        uint32_t shape_len = static_cast<uint32_t>(shape.size());
        ofs.write(reinterpret_cast<const char *>(&shape_len), sizeof(shape_len));
        for (size_t dim : shape)
        {
            ofs.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
        }
    }

    // 5. 写入所有链接信息（直接记录突触原始信息，不转换为矩阵）
    for (size_t i = 0; i < links.size(); i++)
    {
        // 写入链接头部（原逻辑不变：源/目标/激活函数）
        ofs << "[Link:src=" << *layers.find_key(links[i]->source()) << ",tgt=" << *layers.find_key(links[i]->target())
            << ",activation="
            << links[i]->activate()
            << "]\n";
        ofs
            << "Type=Dense\n"; // 标记为全连接层（原逻辑不变）

        // --------------------------
        // 核心修改：写入突触原始信息（替换原矩阵/向量逻辑）
        // --------------------------
        ofs
            << "Synapses=  // 标记突触列表开始\n";
        // 遍历每个突触，写入原始字段（fromIdx、toIdx、weight、bias）
        for (const auto &syn : links[i]->synapses())
        {
            ofs << syn.fromIdx << "," // 源神经元索引
                << syn.toIdx << ","   // 目标神经元索引
                << syn.weight << ","  // 权重
                << syn.bias << "\n";  // 偏置
        }
        ofs
            << "EndSynapses  // 标记突触列表结束\n";
        ofs
            << "\n"; // 链接部分结束，留空行分隔下一个链接
    }
    // 6. 关闭文件
    ofs.close();
    std::cout << "模型已保存到: " << final_filename << std::endl;
}

void normalInitSynapses(std::vector<Synapse> &syns)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::normal_distribution<double> dist(0.0, 1.0);

    for (auto &s : syns)
    { // ✅ 引用！
        s.weight = dist(gen);
        s.bias = dist(gen); // bias 通常初始化为 0，或单独处理
    }
}

bool loadSamples(std::string path, SampleSet &samples)
{
    std::fstream file(path);
    if (!file.is_open())
    {
        std::cout << "loadingSamplesError: " << "couldn't open file " << path << std::endl;
        return 0;
    }
    samples.clear();
    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty())
            continue;
        std::stringstream ss(line);
        std::vector<std::string> tokens;
        std::string token;
        while (ss >> token)
        {
            tokens.push_back(token);
        }
        if (tokens.size() != samples.featureSize + samples.labelSize)
            continue;
        std::vector<double> numTokens;
        {
            bool errorFlag = 0;
            for (int i = 0; i < samples.featureSize + samples.labelSize; i++)
            {
                try
                {
                    numTokens.push_back(std::stod(tokens.at(i)));
                }
                catch (const std::invalid_argument &e)
                {
                    std::cerr << "无效参数: " << e.what() << std::endl;
                    errorFlag = 1;
                    break;
                }
                catch (const std::out_of_range &e)
                {
                    std::cerr << "超出范围: " << e.what() << std::endl;
                    errorFlag = 1;
                    break;
                }
            }
            if (errorFlag)
                continue;
        }
        Sample sp;
        sp.features = std::vector<double>(numTokens.begin(), numTokens.begin() + samples.featureSize);
        sp.labels = std::vector<double>(numTokens.begin() + samples.featureSize, numTokens.end());
        samples.push_back(sp);
    }
    if (samples.size())
        return 1;
    else
    {
        std::cout << "file: " << path << " has no line match sample" << std::endl;
        return 0;
    }
}

void printSampleSet(SampleSet sampleSet)
{
    if (sampleSet.size() <= 20)
        for (int i = 0; i < sampleSet.size(); i++)
        {
            if (sampleSet.featureSize > 30)
            {
                std::cout << "features: ";
                for (int j = 0; j < 30; j++)
                {
                    std::cout << sampleSet.at(i).features.at(j) << " ";
                }
                std::cout << "... ";
            }
            else
            {
                std::cout << "features: ";
                for (int j = 0; j < sampleSet.featureSize; j++)
                {
                    std::cout << sampleSet.at(i).features.at(j) << " ";
                }
            }
            if (sampleSet.labelSize > 10)
            {
                std::cout << "labels: ";
                for (int j = 0; j < 10; j++)
                {
                    std::cout << sampleSet.at(i).labels.at(j) << " ";
                }
                std::cout << "... ";
            }
            else
            {
                std::cout << "labels: ";
                for (int j = 0; j < sampleSet.labelSize; j++)
                {
                    std::cout << sampleSet.at(i).labels.at(j) << " ";
                }
            }
            std::cout << std::endl;
        }
    else
    {
        for (int i = 0; i < 20; i++)
        {
            if (sampleSet.featureSize > 30)
            {
                std::cout << "features: ";
                for (int j = 0; j < 30; j++)
                {
                    std::cout << sampleSet.at(i).features.at(j) << " ";
                }
                std::cout << "... ";
            }
            else
            {
                std::cout << "features: ";
                for (int j = 0; j < sampleSet.featureSize; j++)
                {
                    std::cout << sampleSet.at(i).features.at(j) << " ";
                }
            }
            if (sampleSet.labelSize > 10)
            {
                std::cout << "labels: ";
                for (int j = 0; j < 10; j++)
                {
                    std::cout << sampleSet.at(i).labels.at(j) << " ";
                }
                std::cout << "... ";
            }
            else
            {
                std::cout << "labels: ";
                for (int j = 0; j < sampleSet.labelSize; j++)
                {
                    std::cout << sampleSet.at(i).labels.at(j) << " ";
                }
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for (int k = 0; k < 3; k++)
            std::cout << '.' << std::endl;
    }
}

double sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

double sigmoidDeri(double x)
{
    return x * (1.0 - 1.0 / (1.0 + std::exp(-x)));
}

double linear(double x)
{
    return x;
}

double linearDeri(double x)
{
    return 0;
}

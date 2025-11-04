#include "net.hpp"
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <numeric>
// bool net::loadModel(std::string)
// {
//     // std::cout<<"CialloWorld";
//     std::fstream file("../data/debug.csv");
//     if (!file.is_open())
//         std::cout << "file not open" << std::endl;
//     else
//     {
//         std::cout << "file is opened" << std::endl;
//         std::vector<std::string> fileLines;
//         std::vector<std::vector<std::string>> tokens;
//         std::string line;
//         while (std::getline(file, line))
//         {
//             if (!line.empty())
//             {
//                 fileLines.push_back(line);
//             }
//         }
//         if (!fileLines.empty())
//         {
//             std::cout << "fileLines Length: " << fileLines.size() << std::endl;
//             std::istringstream iss(fileLines.front());
//             std::string token;
//             std::vector<std::string> tokensOfOne;
//             while (iss >> token)
//             {
//                 tokensOfOne.push_back(token);
//             }
//             tokens.push_back(tokensOfOne);
//             if ((!tokens.empty()) && (!tokens.front().empty()))
//                 std::cout << "string token: " << tokens.front().front() << std::endl;
//             int num;
//             try
//             {
//                 num = std::stod(tokens.front().front());
//             }
//             catch (const std::invalid_argument &e)
//             {
//                 std::cerr << "stod无效参数: " << e.what() << std::endl;
//             }
//             catch (const std::out_of_range &e)
//             {
//                 std::cerr << "stod超出范围: " << e.what() << std::endl;
//             }
//             std::cout << "num token: " << num << std::endl;
//         }
//         else
//             std::cout << "fileLines empty" << std::endl;
//     }
// }

void Layer::computeStrides(std::vector<size_t> shp)
{
    size_t sz = shape.size();
    strides.resize(sz);
    strides.back() == 1;
    for (int i = 1; i < sz; i++)
    {
        strides[sz - i - 1] = strides[sz - i] * shape[sz - i];
    }
}

Layer::Layer(std::vector<size_t> shp)
{
    shape = shp;
    computeStrides(shp);
    size_t nr_sz = std::accumulate(shp.begin(), shape.end(), 1ULL, [](size_t a, size_t b)
                                   { return a * b; });
    nuerons.resize(nr_sz);
}

Link::Link(Layer &src, Layer &tgt, std::function<double(double)> atv = [](double v)
                                   { return v; }) : sourceLayer(src), targetLayer(tgt)
{
    initSynapses();
}
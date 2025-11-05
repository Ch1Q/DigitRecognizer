#include "sstream"
#include "net.h"
#include <iostream>
#include <fstream>
int main()
{
    auto in = std::make_shared<Layer>(std::vector<size_t>({2, 2}));
    auto out = std::make_shared<Layer>(std::vector<size_t>({1}));
    auto hid = std::make_shared<Layer>(std::vector<size_t>({3, 3}));

    auto in2hid = std::make_shared<DenseLink>(in, hid);
    in2hid->normalInitSynapses();
    auto hid2out = std::make_shared<DenseLink>(hid, out);

    hid2out->normalInitSynapses();
    Network net(in, out);
    net.addLayer("hid", hid);
    net.addLink(in2hid);
    net.addLink(hid2out);
    net.saveModel("../models/inHidOutDenseLink.nll");
}
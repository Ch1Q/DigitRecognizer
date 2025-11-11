
#include "test.h"
#include "net.h"
#include <iostream>

void testLoadSample()
{
    SampleSet smpSet(256, 10);
    loadSamples("../data/debug.csv", smpSet);
    printSampleSet(smpSet);
}

void testSavingModel()
{
    /*auto in = std::make_shared<Layer>(std::vector<size_t>({2, 2}));
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
    net.saveModel("../models/inHidOutDenseLink.nll");*/
}

void testPredict()
{
    auto in = std::make_shared<Layer>(std::vector<size_t>({1, 5}));
    auto out = std::make_shared<Layer>(std::vector<size_t>({1}));
    auto hid = std::make_shared<Layer>(std::vector<size_t>({3, 3}));

    auto in2hid = std::make_shared<DenseLink>(in, hid);
    // in2hid->normalInitSynapses();
    in2hid->valueInitSynapses(1);
    auto hid2out = std::make_shared<DenseLink>(hid, out);
    // hid2out->normalInitSynapses();
    hid2out->valueInitSynapses(1);

    Network net(in, out);
    net.addLayer(hid);
    net.addLink(in2hid);
    net.addLink(hid2out);

    Sample inputSample;
    inputSample.features.resize(5);
    inputSample.features = {1, 1, 1, 1, 1};
    inputSample.labels.resize(1);

    Sample outputSample = net.predict(inputSample);
    std::cout << outputSample.labels.at(0);
}
#include "mlp_classifier.h"

#include <boost/asio/thread_pool.hpp>
#include <boost/program_options.hpp>
#include <iostream>
#include <numeric>

namespace po = boost::program_options;

int main([[maybe_unused]]int argc, [[maybe_unused]]char** argv) {

    //processing сommand line argument
    po::options_description desc {"Options"};
    desc.add_options()
            ("help,h", "Inference ML for classifying images of wardrobe items by type")
            ("testData,d", po::value<std::string>() -> default_value("./test.csv")
                                            , "the file with test data (data separator in file ',')")
            ("modelDir,m", po::value<std::string>() -> default_value("./model/")
                                            , "path to the dir with files for model. Mlp needs w1.txt, "
                                            "w2.txt files with neuron layer coefficients");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    try {
        std::vector<TestData> data = readTestData(vm["testData"].as<std::string>().c_str());
        std::string w1 = vm["modelDir"].as<std::string>() + "w1.txt";
        std::string w2 = vm["modelDir"].as<std::string>() + "w2.txt";
        MlpClassifier mlp(w1.c_str(), w2.c_str());
        int count = std::accumulate(data.begin(), data.end(), 0
                                    , [&mlp](auto count, auto& val) {
                                        if (val.id == mlp.predict(val.data))
                                            count++;
                                        return count;
                                    });
        double accuracy = (count + .0)/data.size();
        std::cout << accuracy << std::endl;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}
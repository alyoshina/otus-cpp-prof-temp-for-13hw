#pragma once

#include "classifier.h"

#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <iterator>
#include <stdexcept>

using matrix_t = Eigen::MatrixXf;
using vector_t = Eigen::VectorXf;

matrix_t readCoefs(const char* fileName);

class MlpClassifier : public IClassifier {
public:
    MlpClassifier(const char* w1FileName, const char* w2FileName);
    MlpClassifier(const matrix_t& w1, const matrix_t& w2);
    ~MlpClassifier() = default;

    std::size_t numClasses() const override;
    std::size_t predict(const features_t&) const override;
    probas_t predictProba(const features_t&) const override;
private:
    matrix_t w1, w2;
    template<typename T>
    auto sigma(T x) const { return 1/(1 + std::exp(-x)); }
    vector_t sigmav(const vector_t& v) const;
    vector_t softmax(const vector_t& v) const;
};

struct TestData {
    int id;
    MlpClassifier::features_t data;
};

struct DataForRead : public std::string { };
std::istream& operator >> (std::istream& in, DataForRead& output);

struct DataForReadСomma : public std::string { };
std::istream& operator >> (std::istream& in, DataForReadСomma& output);

template <typename T = DataForReadСomma>
std::vector<TestData> readTestData(const char* fileName) {
    std::vector<TestData> result;
    std::ifstream istrm(fileName, std::ifstream::in);
    if (!istrm.is_open()) {
        std::cout << "failed to open " << fileName << std::endl;
        throw std::runtime_error("error read data");
    } else {
        std::for_each((std::istream_iterator<DataForRead>(istrm))
                , std::istream_iterator<DataForRead>()
                , [&result] (auto& val) {
                        std::istringstream iss(val);
                        TestData data;
                        data.id = std::stoi(*(std::istream_iterator<T>(iss)));
                        std::transform((std::istream_iterator<T>(iss))
                            , std::istream_iterator<T>()
                            , std::back_inserter(data.data)
                            , [] (auto& val) { return std::stoi(val); });
                        result.emplace_back(std::move(data));
                    });
    }
    return result;
}
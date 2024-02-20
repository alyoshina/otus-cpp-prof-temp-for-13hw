#include "mlp_classifier.h"

#include <algorithm>
#include <iterator>

std::istream& operator >> (std::istream& in, DataForRead& output) {
    std::getline(in, output);
    return in;
}

matrix_t readCoefs(const char* fileName) {
    std::ifstream istrm(fileName, std::ifstream::in);
    if (!istrm.is_open()) {
        std::cout << "failed to open " << fileName << std::endl;
        throw std::runtime_error("error read coefs");
        return matrix_t(0, 0);
    } else {
        std::size_t rows = 0;
        std::vector<MlpClassifier::feature_t> result;
        std::for_each((std::istream_iterator<DataForRead>(istrm))
                        , std::istream_iterator<DataForRead>()
                        , [&result, &rows] (auto& val) {
                                rows++;
                                std::istringstream iss(val);
                                std::transform((std::istream_iterator<MlpClassifier::feature_t>(iss))
                                            , std::istream_iterator<MlpClassifier::feature_t>()
                                            , std::back_inserter(result)
                                            , [] (auto& val) { return val; });
                            });
        return Eigen::Map<matrix_t>(result.data(), result.size() / rows, rows);
    }
}

MlpClassifier::MlpClassifier(const char* w1FileName
                            , const char* w2FileName)
                        : w1{readCoefs(w1FileName)}
                        , w2{readCoefs(w2FileName)} {}

MlpClassifier::MlpClassifier(const matrix_t& w1
                            , const matrix_t& w2)
                        : w1{w1} , w2{w2} {}

size_t MlpClassifier::numClasses() const {
    return w2.cols();
}

size_t MlpClassifier::predict(const features_t& feat) const {
    auto proba = predictProba(feat);
    auto argmax = std::max_element(proba.begin(), proba.end());
    return std::distance(proba.begin(), argmax);
}

MlpClassifier::probas_t MlpClassifier::predictProba(const features_t& feat) const {
    vector_t x{feat.size()};
    for (size_t i = 0; i < feat.size(); ++i) {
        x[i] = feat[i] / 255;
    }

    auto o1 = sigmav(w1 * x);
    auto o2 = softmax(w2 * o1);

    probas_t res;
    for (size_t i = 0; i < o2.rows(); ++i) {
        res.push_back(o2(i));
    }
    return res;
}

vector_t MlpClassifier::sigmav(const vector_t& v) const {
    vector_t res{v.rows()};
    for (size_t i = 0; i < v.rows(); ++i) {
        res(i) = sigma(v(i));
    }
    return res;
}

vector_t MlpClassifier::softmax(const vector_t& v) const {
    vector_t res{v.rows()};
    float denominator = 0.0f;

    for (size_t i = 0; i < v.rows(); ++i) {
        denominator += std::exp(v(i));
    }
    for (size_t i = 0; i < v.rows(); ++i) {
        res(i) = std::exp(v(i))/denominator;
    }    
    return res;
}

std::istream& operator >> (std::istream& in, DataForReadСomma& output) {
    std::getline(in, output, ',');
    return in;
}
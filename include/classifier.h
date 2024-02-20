#pragma once

#include <cstddef>
#include <vector>

class IClassifier {
public:
    using feature_t = float;
    using features_t = std::vector<feature_t>;
    using probas_t = std::vector<feature_t>;

    virtual ~IClassifier() = default;
    virtual std::size_t numClasses() const = 0;
    virtual std::size_t predict(const features_t&) const = 0;
    virtual probas_t predictProba(const features_t&) const = 0;
};
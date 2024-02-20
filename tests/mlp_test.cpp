#include "mlp_classifier.h"

#include <gtest/gtest.h>
#include <stdio.h>

TEST(Mlp_test, Mlp) {
    std::vector<TestData> data = readTestData<std::string>("data/test_data_mlp.txt");
    ASSERT_TRUE(data.size());
    MlpClassifier mlp("data/model/mlp/coefficients/w1.txt", "data/model/mlp/coefficients/w2.txt");
    std::for_each(data.begin(), data.end(), [&mlp] (auto& val) {
        ASSERT_EQ(val.id, mlp.predict(val.data));
    });
}
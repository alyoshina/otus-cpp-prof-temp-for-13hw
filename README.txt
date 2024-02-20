image sdukshis/cppml

sudo docker build --tag builder .

Docker run:
docker run --rm --name builder -ti -v$(pwd):/usr/src/app builder


cmake .. -DCMAKE_BUILD_TYPE=Release

Run:
./build/bin/fashio_mnist_test
./build/bin/fashio_mnist -d "data/test.csv" -m "data/model/mlp/coefficients/"
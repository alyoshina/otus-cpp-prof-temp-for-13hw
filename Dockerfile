FROM sdukshis/cppml as builder

RUN apt-get update -y && apt-get install libgtest-dev libboost-program-options-dev -y


#RUN apt-get update && \
#    apt-get -y install git python libgtest-dev libboost-program-options-dev

#RUN cd /opt &&\
#    git clone https://github.com/catboost/catboost.git

#RUN cd /opt/catboost && \
#    chmod +x ya && \
#    ./ya make -r catboost/libs/model_interface

#FROM sdukshis/cppml

#COPY --from=builder /opt/catboost/catboost/libs/model_interface/libcatboostmodel.so /usr/local/lib/
#RUN cd /usr/local/lib/ && ln -s libcatboostmodel.so libcatboostmodel.so.1
#COPY --from=builder /opt/catboost/catboost/libs/model_interface/*.h /usr/local/include/catboost/

#RUN curl -s https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.5.0.tar.gz | tar -C /usr/local -xzf - && \
#    ldconfig


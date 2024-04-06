
FROM tensorflow/tensorflow:1.14.0-gpu

RUN mkdir /app
WORKDIR /app

COPY --chown=root ./system-repositories-linux.txt /app
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update -y && \
    apt-get install -y software-properties-common;
RUN xargs -L1 add-apt-repository <./system-repositories-linux.txt

COPY --chown=root ./system-requirements-linux.txt /app
RUN apt-get update -y && \
    xargs apt-get install -y <./system-requirements-linux.txt

COPY --chown=root ./requirements.txt /app
RUN python3.7 -m pip install --upgrade pip && \
    python3.7 -m pip install -r requirements.txt

RUN curl https://sumo.dlr.de/releases/1.19.0/sumo-src-1.19.0.tar.gz \
    | tar -xz
ENV SUMO_HOME=sumo-1.19.0
RUN mkdir sumo-1.19.0/build/cmake-build && cd sumo-1.19.0/build/cmake-build \
    && cmake ../.. && make -j$(nproc)

ENV SOURCE_CODE_PATH=/app/src

ENV SCENARIO_PATH=/app/scenario
ENV CONFIG_PATH=/app/config

ENV USED_SCENARIO_PATH=/app/output/traffic_management/data
ENV USED_CONFIG_PATH=/app/output/traffic_management/config
ENV DATA_PATH=/app/output/traffic_management/data
ENV MODEL_PATH=/app/output/traffic_management/model
ENV RECORDS_PATH=/app/output/traffic_management/records
ENV SUMMARY_PATH=/app/output/traffic_management/summary
ENV METRIC_PATH=/app/output/traffic_management/metric

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user && adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
RUN chown -R $USER_ID:$GROUP_ID /app && chmod -R a=rwx /app

USER user

RUN mkdir -p ${USED_SCENARIO_PATH} ${USED_CONFIG_PATH} ${DATA_PATH} ${MODEL_PATH} ${RECORDS_PATH} ${SUMMARY_PATH} ${METRIC_PATH}

VOLUME ${SOURCE_CODE_PATH}

VOLUME ${SCENARIO_PATH}
VOLUME ${CONFIG_PATH}

VOLUME ${USED_SCENARIO_PATH}
VOLUME ${USED_CONFIG_PATH}
VOLUME ${DATA_PATH}
VOLUME ${MODEL_PATH}
VOLUME ${RECORDS_PATH}
VOLUME ${SUMMARY_PATH}
VOLUME ${METRIC_PATH}


ENTRYPOINT ["python3.7"]
CMD ["src/simulation.py"]

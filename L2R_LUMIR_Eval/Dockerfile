FROM python:3.9-slim

RUN apt-get update && apt-get -y install build-essential git


RUN groupadd -r evaluator && useradd -m --no-log-init -r -g evaluator evaluator

RUN mkdir -p /opt/evaluation /input /output \
    && chown evaluator:evaluator /opt/evaluation /input /output 

USER evaluator
WORKDIR /opt/evaluation

ENV PATH="/home/evaluator/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

COPY --chown=evaluator:evaluator ground-truth /opt/evaluation/ground-truth
COPY --chown=evaluator:evaluator requirements.txt /opt/evaluation/
RUN python -m pip install --user -r requirements.txt

COPY --chown=evaluator:evaluator evaluation.py /opt/evaluation/
COPY --chown=evaluator:evaluator utils.py /opt/evaluation/
COPY --chown=evaluator:evaluator digital_diffeomorphism.py /opt/evaluation/
#COPY --chown=evaluator:evaluator surface_distance /opt/evaluation/

RUN git clone -n https://github.com/deepmind/surface-distance.git
RUN cd surface-distance && git checkout ee651c8
RUN python -m pip install --user -e surface-distance

ENTRYPOINT "python" "-m" "evaluation" "-c1" "ground-truth/LUMIR_VAL_Landmark_evaluation_config.json" "-c2" "ground-truth/LUMIR_VAL_Segmentation_evaluation_config.json" "-i" "/input/" "-d" "/opt/evaluation/ground-truth/" "-o" "/output/metrics.json"

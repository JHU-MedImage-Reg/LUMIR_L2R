FROM python:3.8-slim

RUN python -m pip install --user -U pip

RUN mkdir -p /input /output /json_dir

COPY ./requirements.txt .
COPY ./infer_script.py .

RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
RUN python3 -m pip install --user -r requirements.txt

ENTRYPOINT ["python3","-u","./infer_script.py"]
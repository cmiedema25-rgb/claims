FROM python:3.12-slim
RUN pip install --no-cache-dir numpy==2.3.5 scipy==1.17.0
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
WORKDIR /app
COPY data /app/data
CMD ["sleep", "infinity"]

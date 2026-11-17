# CPU container for the tote depalletizing pipeline (CAD stand-in, lidar/camera simulator, detection, ICP, studies).
#   docker build -t tote-depalletizing .
#   docker run --rm -v $PWD/results:/app/results tote-depalletizing python scripts/grid_icp_detect.py --scans data/sim --out results/grid_p2mesh
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libegl1 libgomp1 libx11-6 libglib2.0-0 libusb-1.0-0 libudev1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1 MPLBACKEND=Agg
CMD ["bash", "-c", "python scripts/simulate_scans.py --picks 4 && python scripts/run_pipeline.py && python scripts/grid_icp_detect.py"]

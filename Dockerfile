FROM python:3.10

WORKDIR /app

RUN python -m pip cache purge

COPY requirements_shared.txt .
COPY requirements_specific.txt .

RUN pip install --no-cache-dir -r requirements_shared.txt
RUN pip install --no-cache-dir -r requirements_specific.txt

# Install Jupyter
RUN pip install jupyter

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Run Jupyter Notebook when the container launches
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
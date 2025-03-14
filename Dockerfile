FROM jupyter/minimal-notebook

WORKDIR /app

COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# The default command to start Jupyter
CMD ["start-notebook.sh"]

# Use official Python image
FROM python:3.11

# Set working directory inside the container
WORKDIR /app

# Copy everything from the local directory to the container
COPY . /app

# Install dependencies (Ensure specific numpy & scikit-learn versions for compatibility)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt numpy==1.24.3 scikit-learn==1.3.0 

# Expose ports (Flask API on 5000, Streamlit UI on 8501)
EXPOSE 5000 8501

# Run both Flask API and Streamlit UI together
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:5000 main:app & streamlit run app.py --server.port 8501 --server.address 0.0.0.0"]

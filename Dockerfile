# Pull base image with Python
FROM python:3.9.16-slim-bullseye

# Fixing zlib and pcre2 vulnerabilities
RUN apt update
RUN apt install zlib1g
RUN apt install libpcre2-8-0
RUN apt install python3-pip -y

# Create a working directory for our container
WORKDIR /code

# Copy the requirements into the code folder
COPY requirements.txt /code/requirements.txt

# Install the packages and the spacy package
RUN pip cache purge
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /code/requirements.txt
RUN python -m spacy download en_core_web_lg
# Install other requirements
RUN pip install fastapi uvicorn transformers==4.19.2 torch==1.13.1
RUN pip install scikit-learn==1.1.1
RUN pip install beautifulsoup4==4.11.1
RUN pip install boto3==1.24.31

# Copy all the code files into the app subfolder
COPY . /code/app

# Expose port 8000 which we are hosting the app on
EXPOSE 8000

# Change our work directory to the app subfolder
WORKDIR /code/app

# Add user so we can run the container as the non-root user
RUN useradd -u 9624 shaun
USER shaun

# Execute command for hosting our API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
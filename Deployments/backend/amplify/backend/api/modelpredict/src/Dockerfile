FROM python:3.9

WORKDIR /code
COPY requirements.txt /code/requirements.txt
RUN  pip install --no-cache-dir -r /code/requirements.txt

EXPOSE 80
COPY ./app /code/app

WORKDIR /code/app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
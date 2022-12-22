FROM python:3.10
RUN apt-get update -yq && apt-get upgrade -yq && apt-get install libgl1 -yq
COPY ./app /docker_app
COPY requirements.txt /
RUN pip install -r requirements.txt
WORKDIR /docker_app
CMD ["flask", "--app", "app", "--debug", "run", "--host=0.0.0.0"]
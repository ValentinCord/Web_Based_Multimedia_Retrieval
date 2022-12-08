FROM python:3.10
RUN apt-get update -yq && apt-get upgrade -yq
COPY ./app /docker_app
COPY requirements.txt /
WORKDIR /docker_app
CMD ["flask", "--app", "app", "run", "--host=0.0.0.0"]
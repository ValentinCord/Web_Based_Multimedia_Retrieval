# **Virtual Environment Setup**
```console
sudo apt-get install -y python3-venv
python3 -m venv env
source env/bin/activate
pip install flask
python -m pip freeze > requirements.txt
```

# **Build Docker**
```console
docker build -t [IMAGE_NAME] .
```

# **Run Docker**
```console
docker run -p 5000:5000 [IMAGE_NAME]
```

# **Clean Docker**
```console
docker system prune -a
```

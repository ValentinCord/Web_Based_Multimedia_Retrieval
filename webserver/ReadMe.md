# **Docker Installation on Ubuntu**
```console
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo 
apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/
linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install -y docker-ce
sudo systemctl status docker
```

# **Add User to Docker Group**
```console
sudo usermod -aG docker [USER_NAME]
```

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

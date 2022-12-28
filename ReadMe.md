# **Docker Installation on Ubuntu**
```console
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo 
$ apt-key add -
$ sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/
$ linux/ubuntu $(lsb_release -cs) stable"
$ sudo apt-get update
$ sudo apt-get install -y docker-ce
$ sudo systemctl status docker
$ docker --version
```

# **Docker Compose Installation on Ubuntu**
```console
$ sudo curl -L "https://github.com/docker/compose/rel... -s)-$(uname -m)" -o /usr/local/bin/docker-compose
$ sudo chmod +x /usr/local/bin/docker-compose
$ docker-compose --version
```

# **Docker Compose Utilization**
```console
$ docker-compose build
$ docker-compose up
$ docker-compose down
```




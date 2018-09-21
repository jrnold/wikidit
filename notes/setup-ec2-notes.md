```console
$ sudo apt-get update && apt-get upgrade -y
```

```console
$ sudo apt-get install build-essential
```

## Node

```console
$ curl -sL https://deb.nodesource.com/setup_8.x | sudo -E bash -
$ sudo apt-get install -y nodejs npm
$ curl -sL https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add -
$ echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list
$ sudo apt-get update && sudo apt-get install yar
```

## Docker

```console
sudo apt-get install docker docker-compose
```

## Conda

```console
wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
chmod +x Anaconda3-5.2.0-Linux-x86_64.sh
./Anaconda3-5.2.0-Linux-x86_64.sh
rm ./Anaconda3-5.2.0-Linux-x86_64.sh
```

## zsh

Install ZSH shell and other setup
```console
$ sudo apt-get install zsh zsh-doc zsh-antigen zsh-theme-powerlevel9k zsh-syntax-highlighting
$ sudo chsh -s /usr/bin/zsh $(whoami)
```

## Nuclide Watchman

```console
$ sudo apt-get install libssl-dev
$ git clone https://github.com/facebook/watchman.git
$ cd watchman
$ git checkout v4.9.0  # the latest stable release
$ ./autogen.sh
$ ./configure
$ make
$ sudo make install
```

Create SSH Key

```console
$ ssh-keygen -t rsa -b 4096 -C 'ubuntu@jaqobi AWS'
```

# git configure
```
$ git configure --global user.name "jrnold"
$ git configure --global user.email "jeffrey.arnold@gmail.com"
```

Add SSH key to github

## Nuclide Remote

https://nuclide.io/docs/features/remote/

```console
$ sudo npm install -g nuclide
```

# Associate Elastic IP with an instance

# ubuntu
FROM ubuntu:24.04

# get the latest
RUN apt dist-upgrade -y \
    # update the package repository
    && apt update \
    # install general purpose packages
    && apt install -y --no-install-recommends \
    curl \
    git \
    git-core \
    bash-completion \
    openssh-server \
    make \
    nano \
    vim \
    sudo \
    ca-certificates

# install python and pip
RUN apt install -y --no-install-recommends \
    libpython3-dev \
    python3-venv \
    python3-pip

# to avoid user interaction with tzdata asking for timezone
ARG DEBIAN_FRONTEND=noninteractive

# environment
# colorize (for fun)
ENV TERM=xterm-256color

# global args
ARG user=user
ARG home=/home/${user}

# setup user
ENV USER=${user}
ENV HOME=${home}
RUN useradd -s /bin/bash --create-home --home-dir $HOME $USER \
    && usermod -aG sudo $USER
RUN echo "$USER:password" | chpasswd
WORKDIR $HOME
USER $USER
# the startup file
COPY --chown=$USER:$USER bashrc .bashrc
# copy the vscode settings
WORKDIR ${HOME}/.vscode
COPY --chown=$USER:$USER settings.json settings.json

# default user and work directory when running the container
USER ${USER}
WORKDIR ${home}
# remove annoying message on how to run a sudo command
RUN touch ~/.sudo_as_admin_successful
# create .config directory
RUN mkdir $HOME/.config
# the git configuration files
COPY --chown=${USER}:${USER} gitconfig $HOME/.gitconfig

# end of file

FROM rapidsai/rapidsai-core:22.10-cuda11.5-base-ubuntu20.04-py3.9

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update
RUN apt-get install -y ssh
RUN apt-get install -y vim

WORKDIR /app

CMD [ "python"]

# ssh setting
RUN echo 'root:8652' | chpasswd
RUN sed -i "s/#Port.*/Port 22/" /etc/ssh/sshd_config && \
    sed -i "s/#PermitRootLogin.*/PermitRootLogin yes/" /etc/ssh/sshd_config && \
    sed -i "s/#PasswordAuthentication.*/PasswordAuthentication yes/" /etc/ssh/sshd_config
  
ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile
  
EXPOSE 22

# set entrypoint to restart ssh automatically
ENTRYPOINT service ssh restart && bash
FROM nginx

RUN apt update

RUN apt install -y python3 python3-pip

RUN apt install -y git

RUN apt-get -y install fcgiwrap

COPY ./docker/fcgiwrap.conf /etc/nginx/fcgiwrap.conf

COPY ./docker/default.conf /etc/nginx/conf.d/default.conf

WORKDIR /gitfolder

COPY ./sources /gitfolder/instance_seg_abris/

RUN apt install -y $(cat ./instance_seg_abris/requirements.system)

RUN pip3 install -r ./instance_seg_abris/requirements.txt --break-system-packages

USER root

RUN chmod 777 /gitfolder/instance_seg_abris/main.py

RUN mkdir /data

COPY ./docker/docker-entrypoint.sh /docker-entrypoint.sh

RUN chmod +x /docker-entrypoint.sh

RUN service nginx restart

ENTRYPOINT ["/docker-entrypoint.sh"]

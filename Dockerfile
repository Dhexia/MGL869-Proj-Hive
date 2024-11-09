FROM jupyter/base-notebook:latest
WORKDIR /Hive
USER root

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y python3-pip git
RUN pip3 install pandas GitPython requests

COPY docker-startup.sh /usr/local/bin/docker-startup.sh
RUN chmod +x /usr/local/bin/docker-startup.sh
RUN apt-get install -y libglib2.0-0

EXPOSE 8888

ENTRYPOINT ["/usr/local/bin/docker-startup.sh"]



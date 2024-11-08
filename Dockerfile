FROM jupyter/base-notebook:latest
WORKDIR /Hive
USER root

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y python3-pip git
RUN pip3 install pandas GitPython requests

COPY UnderstandArchive/Understand-CLI-6.5.1160-Linux-64bit.tgz /temp/Understand-CLI-6.5.1160-Linux-64bit.tgz

RUN tar -xvzf /temp/Understand-CLI-6.5.1160-Linux-64bit.tgz -C /Hive

# Expose port for Jupyter
EXPOSE 8888

CMD ["jupyter", "notebook","--allow-root", "--no-browser", "--NotebookApp.token='my-token'"]
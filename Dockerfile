# our base image
FROM continuumio/miniconda3

# Create the environment:
COPY environment.yml /usr/src/app/
RUN conda env create -f /usr/src/app/environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "test", "/bin/bash", "-c"]

# install Python modules needed by the Python app
COPY requirements.txt /usr/src/app/
RUN pip install --no-cache-dir -r /usr/src/app/requirements.txt

# copy files required for the app to run
COPY test/app.py /usr/src/app/test/
COPY test/templates/index.html /usr/src/app/test/templates/

# tell the port number the container should expose
EXPOSE 5000

# run the application
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "test", "python", "/usr/src/app/test/app.py"]
# Use an official Python runtime as a parent image
FROM astrabert/everything-ai

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

#Upgrade gradio
RUN pip install gradio_molecule3d

# Expose the port that the application will run on
EXPOSE 8760

ENTRYPOINT [ "python3", "select_and_run.py" ]

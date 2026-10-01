FROM quay.io/jupyterhub/jupyterhub:5.5.1

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Not under /srv/jupyterhub: the volume mounted there would shadow every rebuilt config.
COPY jupyterhub_config.py /etc/jupyterhub/jupyterhub_config.py

WORKDIR /srv/jupyterhub

CMD ["jupyterhub", "-f", "/etc/jupyterhub/jupyterhub_config.py"]

"""
JupyterHub configuration for GRAPHIT with DockerSpawner: one container per student from the single-user image.

- ``hub_connect_ip`` must match the hub's ``container_name`` in docker-compose.yml, and
  ``network_name`` must be set explicitly, because spawned containers are not managed by
  Compose; otherwise the spawn runs into ``http_timeout``.
- Only ``work/`` is persistent; mounting the whole home would shadow the image's dotfiles.
  JupyterLab workspaces (the extension's state) are moved into the volume for that reason.
- Optional GRAPHIT switches are passed on only if set, so the seeding script can tell
  "unset" from "empty". They replace the hidden settings editor; a hand-edited studentId
  would write the learning state to someone else's node.
- NativeAuthenticator is required because the student id is seeded from
  ``$JUPYTERHUB_USER`` and the backend trusts ``X-Student-Id`` without own authentication.
  Remaining risk: whoever reaches the backend port directly can send any id, so the
  backend is bound to 127.0.0.1. ``allow_all`` is needed since JupyterHub 5; access is
  controlled by ``open_signup = False`` plus admin approval.
"""

import os

import nativeauthenticator

c = get_config()  # noqa: F821

c.JupyterHub.bind_url = "http://:8000"
c.JupyterHub.hub_ip = "0.0.0.0"

c.JupyterHub.hub_connect_ip = os.environ.get("HUB_CONNECT_IP", "jupyterhub")

c.JupyterHub.spawner_class = "dockerspawner.DockerSpawner"
c.DockerSpawner.image = os.environ["DOCKER_SINGLEUSER_IMAGE"]

c.DockerSpawner.network_name = os.environ["DOCKER_NETWORK_NAME"]

c.DockerSpawner.use_internal_ip = True

c.DockerSpawner.remove = True
c.DockerSpawner.debug = True
c.DockerSpawner.notebook_dir = "/home/jovyan/work"

c.DockerSpawner.volumes = {"jupyterhub-user-{username}": "/home/jovyan/work"}

_OPTIONAL = (
    "GRAPHIT_SHOW_DIAGNOSTICS",
    "GRAPHIT_MOCK_MODE",
    "GRAPHIT_STREAMING",
    "GRAPHIT_CHAT_MODEL",
    "GRAPHIT_REVIEW_SESSION_SIZE",
    "GRAPHIT_PERSIST_CHAT_HISTORY",
)
c.DockerSpawner.environment = {
    "GRAPHIT_BASE_URL": os.environ["GRAPHIT_BASE_URL"],
    "JUPYTERLAB_WORKSPACES_DIR": "/home/jovyan/work/.jupyterlab-workspaces",
    **{
        name: os.environ[name].strip()
        for name in _OPTIONAL
        if os.environ.get(name, "").strip()
    },
}

c.Spawner.default_url = "/lab"
c.Spawner.http_timeout = 120
c.Spawner.start_timeout = 180

c.JupyterHub.authenticator_class = "nativeauthenticator.NativeAuthenticator"
c.NativeAuthenticator.open_signup = False
c.NativeAuthenticator.minimum_password_length = 8
c.NativeAuthenticator.check_common_password = True
c.JupyterHub.template_paths = [
    f"{os.path.dirname(nativeauthenticator.__file__)}/templates/"
]

c.Authenticator.admin_users = {
    u.strip() for u in os.environ.get("JUPYTERHUB_ADMIN", "admin").split(",") if u.strip()
}

c.Authenticator.allow_all = True

c.JupyterHub.db_url = "sqlite:////srv/jupyterhub/jupyterhub.sqlite"
c.JupyterHub.cookie_secret_file = "/srv/jupyterhub/jupyterhub_cookie_secret"

c.JupyterHub.shutdown_on_logout = False
c.JupyterHub.cleanup_servers = False

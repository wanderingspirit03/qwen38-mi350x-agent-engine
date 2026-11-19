FROM rocm/atom-dev@sha256:ad256a94209d2f58bfa6b0a69c555e0f8aab4fd8b76673c3532258803599099c
ENV GPU_MAX_HW_QUEUES=1 PYTHONUNBUFFERED=1
WORKDIR /opt/eng
COPY eng/csrc /opt/eng/eng/csrc
RUN chmod -R a+rX /opt/eng && /opt/rocm/bin/hipcc -O3 --offload-arch=gfx950 -shared -fPIC -o /opt/eng/eng/libengk.so eng/csrc/*.hip
COPY eng /opt/eng/eng
COPY tests /opt/eng/tests
RUN chmod -R a+rX /opt/eng

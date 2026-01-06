FROM dolfinx/dolfinx:v0.8.0

RUN pip3 install --no-cache-dir git+https://github.com/FEniCS/fiat.git

RUN pip3 install --no-cache-dir pdbpp

RUN apt update && \
    apt install -y --no-install-recommends --no-install-suggests ccache && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
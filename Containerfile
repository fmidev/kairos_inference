FROM rockylinux/rockylinux:8

RUN rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm \
             https://download.fmi.fi/smartmet-open/rhel/8/x86_64/smartmet-open-release-latest-8.noarch.rpm

RUN dnf -y install dnf-plugins-core && \
    dnf config-manager --set-enabled powertools && \
    dnf config-manager --setopt="epel.exclude=eccodes*" --save && \
    dnf -y --setopt=install_weak_deps=False install python3.11 python3.11-pip python3.11-setuptools eccodes git && \
    dnf -y clean all && rm -rf /var/cache/dnf

RUN git clone https://github.com/fmidev/kairos_inference.git

WORKDIR /kairos_inference

ENV VIS_TAG=0625
ENV CLDBASE_TAG=20250610

ADD https://lake.fmi.fi/ml-models/meps-aerodrome/xgb_vis_0_$VIS_TAG.json /kairos_inference
ADD https://lake.fmi.fi/ml-models/meps-aerodrome/xgb_vis_1_$VIS_TAG.json /kairos_inference
ADD https://lake.fmi.fi/ml-models/meps-aerodrome/xgb_vis_2_$VIS_TAG.json /kairos_inference
ADD https://lake.fmi.fi/ml-models/meps-aerodrome/xgb_cbase_0_36_$CLDBASE_TAG.json /kairos_inference
ADD https://lake.fmi.fi/ml-models/meps-aerodrome/xgb_cbase_4_9_$CLDBASE_TAG.json /kairos_inference

RUN chmod 644 xgb_vis_0_$VIS_TAG.json && \
    chmod 644 xgb_vis_1_$VIS_TAG.json && \
    chmod 644 xgb_vis_2_$VIS_TAG.json && \
    chmod 644 xgb_cbase_0_36_$CLDBASE_TAG.json && \
    chmod 644 xgb_cbase_4_9_$CLDBASE_TAG.json && \
    update-alternatives --set python3 /usr/bin/python3.11 && \
    python3 -m pip --no-cache-dir install -r requirements.txt

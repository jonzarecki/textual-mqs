FROM continuumio/miniconda2

RUN apt update
RUN apt install -y gcc g++ build-essential gfortran libatlas-base-dev liblapacke-dev
COPY requirements.txt /
RUN pip install cython
RUN pip install numpy
RUN pip install libact
RUN pip uninstall -y libact
RUN pip install -e git://github.com/yonilx/libact.git#egg=libact
RUN pip install -r /requirements.txt
RUN pip install -e git://github.com/yonilx/simpleai.git#egg=simpleai

RUN pip install "spacy<2"
RUN python -m spacy download en_core_web_md
RUN echo "import nltk;nltk.download('stopwords')" | python
RUN echo "import nltk;nltk.download('averaged_perceptron_tagger')" | python
RUN echo "import nltk;nltk.download('wordnet')" | python
RUN echo "import nltk;nltk.download('punkt')" | python
RUN pip install "gensim==3.2.*"
RUN pip install "spacy<1.3"

# can run using ssh interpreter or pycharm

RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo 'root:Aa123456' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
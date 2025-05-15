FROM ubuntu:latest

# non interactive frontend for locales
ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true
ENV DIR /diploma-bachelor

RUN mkdir $DIR && \
    echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections

    # Добавляем инструменты для сборки Perl-модулей
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libc6-dev \
    libperl-dev

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    wget \
    curl \
    git \
    make \
    pandoc \
    unzip && \
    apt-get install --no-install-recommends -y \
    texlive-base \
    texlive-latex-extra \
    texlive-extra-utils \
    texlive-xetex \
    texlive-lang-cyrillic \
    texlive-fonts-extra \
    texlive-science \
    texlive-latex-recommended \
    latexmk \
    procps zip \
    biber \ 
    texlive-bibtex-extra \
    locales \
    python3-pygments 

# Установка pip и pygments для поддержки minted
RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip3 install --break-system-packages Pygments



RUN cpan -i App::cpanminus \
    cpanm YAML::Tiny \
    cpanm File::HomeDir 
# Times New Roman and other fonts
# 1. Установка зависимостей
RUN apt-get update && \
    echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections && \
    apt-get install -y --no-install-recommends \
        ttf-mscorefonts-installer \
        fonts-freefont-ttf \
        fontconfig \
        wget \
        unzip

# 2. Установка XITS Math (из GitHub)
RUN mkdir -p /usr/share/fonts/xits && \
    wget -O /usr/share/fonts/xits/XITSMath-Regular.otf \
        https://github.com/aliftype/xits/raw/master/XITSMath-Regular.otf

# 3. Установка PT Mono (из Google Fonts)
# Копируем шрифты из локальной папки в системный каталог
COPY fonts /usr/share/fonts/pts/


# Копируем отдельно PT Mono и PT Sans в подкаталоги
RUN mkdir -p /usr/share/fonts/ptmono /usr/share/fonts/ptsans

COPY fonts/PTMono-Regular.ttf /usr/share/fonts/ptmono/
COPY fonts/PTSans-Regular.ttf /usr/share/fonts/ptsans/
COPY fonts/PTSans-Bold.ttf /usr/share/fonts/ptsans/
COPY fonts/PTSans-Italic.ttf /usr/share/fonts/ptsans/
COPY fonts/PTSans-BoldItalic.ttf /usr/share/fonts/ptsans/

RUN fc-cache -f -v



# generating locales
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8 LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

VOLUME $DIR
WORKDIR $DIR
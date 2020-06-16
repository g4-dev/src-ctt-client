#!/bin/bash

sudo pip install wheel
sudo apt-get -y install build-essential
sudo apt-get -y remove libportaudio2
sudo apt-get -y install libasound2-dev
git clone -b alsapatch https://github.com/gglockner/portaudio
cd portaudio
./configure && make
sudo make install
sudo ldconfig
cd ..
sudo pip3 install pyaudio
sudo cp ./src/transcriber.py /usr/local/bin/transcriber

# fix conf
ALSA_CONF=/usr/share/alsa/alsa.conf
sudo sed -i -e "s/^cards.pcm.rear\s*/#cards.pcm.rear/g" ${ALSA_CONF}
sudo sed -i -e "s/^cards.pcm.center_lfe\s*/#cards.pcm.center_lfe/g" ${ALSA_CONF}
sudo sed -i -e "s/^cards.pcm.side\s*/#cards.pcm.side/g" ${ALSA_CONF}

#pcm.rear
#cards.pcm.rear
#pcm.center_lfe
#cards.pcm.center_lfe
#pcm.side
#cards.pcm.side
#pcm.surround21 cards.pcm.surround21
#pcm.surround40 cards.pcm.surround40
#pcm.surround41 cards.pcm.surround41
#pcm.surround50 cards.pcm.surround50
#pcm.surround51 cards.pcm.surround51
#pcm.surround71 cards.pcm.surround71
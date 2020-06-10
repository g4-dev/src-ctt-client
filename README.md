# src-ctt-client

Sources du model deepspeech pour convertir des audios en texte.

### TODO

- redirect audio :  https://github.com/GeoffreyPlitt/vagrant-audio/blob/master/Vagrantfile

### Environement
Machine virtuelle Basé sur le playbook [playbook-ctt-client](https://github.com/g4-dev/playbook-ctt-client)

### Requis:

- `vagrant` (2.2)
- `git` (git bash / cygwin)
- `vs-code` ou autre IDE légé
- **CPU +2Core** / **4094RAM**

### install

Config ssh si pas déjà fait :

**Ouvrir GIT BASH**, recommencer ces commandes après avoir touché à votre ssh

Si possible editer sa config ssh :

```sh
echo "
StrictHostKeyChecking no
ForwardX11 yes
ForwardAgent yes
" >> ~/.ssh/config
```

```sh
ssh-keygen -t rsa -m PEM -b 4096 -C "your_email@example.com"`
# ajoutez la clé sur github Settings > SSH and GPG keys
eval $(ssh-agent -s)
ssh-add
```

`vagrant up` ou en cas de problème `vagrant reload --provision`

### Outils utilisés

> Préinstallés dans la machine virtuelle
- [deepspeech v0.7](https://deepspeech.readthedocs.io/en/v0.7.1/USING.html)
- [sox]()
- [pyaudio]()

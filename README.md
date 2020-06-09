# src-ctt-client

Sources du model deepspeech pour convertir des audios en texte.

### Environement
Machine virtuelle Basé sur le playbook [playbook-ctt-client](https://github.com/g4-dev/playbook-ctt-client)

### Requis:

- `vagrant` (2.2)
- `git` (git bash / cygwin)
- `vs-code` ou autre IDE légé
- **CPU +2Core** / **4094RAM**

### install

Config ssh si pas déjà fait :

**Ouvrir GIT BASH **, recommencer ces commandes après avoir touché à votre ssh

```sh
ssh-keygen -t rsa -m PEM -b 4096 -C "your_email@example.com"` # ajoutez sur github
eval $(ssh-agent -s)
ssh-add
```

`vagrant up`

### Outils utilisés

> Préinstallés dans la machine virtuelle
- [deepspeech v0.7](https://deepspeech.readthedocs.io/en/v0.7.1/USING.html)
- [sox]()
- [pyaudio]()

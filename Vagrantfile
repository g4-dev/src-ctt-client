# -*- mode: ruby -*-
# vi: set ft=ruby :

require_relative '.manala/vg-facade/facade'

Vagrant.configure('2') do |vagrant|
  config.ssh.forward_x11 = true # useful since some audio testing programs use x11
  Facade.new(vagrant, __dir__)

  # enable audio drivers on VM settings
  config.vm.provider :virtualbox do |vb|
    if RUBY_PLATFORM =~ /darwin/
      vb.customize ["modifyvm", :id, '--audio', 'coreaudio', '--audiocontroller', 'hda'] # choices: hda sb16 ac97
    elsif RUBY_PLATFORM =~ /mingw|mswin|bccwin|cygwin|emx/
      vb.customize ["modifyvm", :id, '--audio', 'dsound', '--audiocontroller', 'ac97']
    end
  end
end

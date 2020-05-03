# -*- mode: ruby -*-
# vi: set ft=ruby :

# TODOC
class Base
	@UPDATE_BOX = ENV['BOX_UPDATE']
	@UPDATE_VBGUEST = ENV['VBGUEST_UPDATE']

	def initialize()
		@UPDATE_BOX = defined?(@UPDATE_BOX) || $config.box_update
		@UPDATE_VBGUEST = defined?(@UPDATE_VBGUEST) || $config.vb_guest_update
		requirements
		# Dispatch
		box
		ssh
	end

	def box()
		$vagrant.vm.hostname = $config.domain
		$vagrant.vm.box = $config.box || "loic-roux-404/deb64-buster"
		$config.box_version ? $vagrant.vm.box_version = $config.box_version : nil
		$vagrant.vm.box_check_update = @UPDATE_BOX
		$vagrant.vbguest.auto_update = @UPDATE_VBGUEST
	end

	def ssh()
		id_rsa_path        = File.join(Dir.home, ".ssh", "id_rsa")
		id_rsa_ssh_key     = File.read(id_rsa_path)
		id_rsa_ssh_key_pub = File.read(File.join(Dir.home, ".ssh", "id_rsa.pub"))
		insecure_key_path  = File.join(Dir.home, ".vagrant.d", "insecure_private_key")
		# set vagrant ssh settings
		$vagrant.ssh.insert_key = false
		$vagrant.ssh.forward_agent = true
		$vagrant.ssh.private_key_path = [id_rsa_path, insecure_key_path]
		# add personal key into vm to assure faster ssh auth
		ssh_path = "/home/vagrant/.ssh"
		$vagrant.vm.provision :shell, 
				inline: "echo '#{id_rsa_ssh_key}' > #{ssh_path}/id_rsa && chmod 600 #{ssh_path}/id_rsa"
		$vagrant.vm.provision :shell, 
				inline: "echo '#{id_rsa_ssh_key_pub}' > #{ssh_path}/authorized_keys && chmod 600 #{ssh_path}/authorized_keys"
	end

	def requirements()
		if @UPDATE_VBGUEST && !Vagrant.has_plugin?('vagrant-vbguest')
			system('vagrant plugin install vagrant-vbguest')
		end
	end
# end Base class
end
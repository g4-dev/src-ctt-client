# -*- mode: ruby -*-
# vi: set ft=ruby :

class Fs < Component  
  PREFIX = 'fs_'

  def initialize()
    super(PREFIX)
    requirements
    @opts = $config.fs.opts
    @host = $config.paths.host
    @guest = $config.paths.guest
    self.send(@PREFIX+$config.fs.type)
    provision_trigger
  end

  def fs_rsync
    $vagrant.vm.synced_folder @host, @guest, 
      disabled: @opts.disabled, 
      type: 'rsync', 
      rsync__auto: @opts.auto,
      rsync__args: ["--archive", "--delete", "--no-owner", "--no-group","-q", "-W"],
      rsync__exclude: @opts.ignored
  end

  def fs_nfs
    # NFS config / bind vagrant user to nfs mount
    if Vagrant::Util::Platform.darwin?
      $vagrant.vm.synced_folder @host, @guest, 
        nfs: true, 
        mount_options: ['rw','tcp','fsc','noatime','rsize=8192','wsize=8192','noacl','actimeo=2'],
        linux__nfs_options: ['rw','no_subtree_check','all_squash','async'],
        disabled: @opts.disabled
      $vagrant.bindfs.bind_folder @guest, @guest
    else
      # linux nfs 4 server
      $vagrant.vm.synced_folder @host, @guest, 
        nfs: true, 
        nfs_version: 4, 
        nfs_udp: false, 
        mount_options: ['rw','noac','actimeo=2','nolock'],
        disabled: @opts.disabled
    end
  end

  def fs_smb
    $vagrant.vm.synced_folder @host, @guest, 
      type: 'smb',
      mount_options: ["vers=2.0"],
      disabled: @opts.disabled
  end

  def fs_vbox
    $vagrant.vm.synced_folder @host, @guest, disabled: @opts.disabled
  end

  def provision_trigger
    # reload shared folder after provision
    if File.exist? $__dir__ + "/.vagrant/machines/default/virtualbox/action_provision"
      $vagrant.trigger.after :provision do |t|
        t.info = "Reboot after provisioning"
        t.run = { :inline => "vagrant reload" }
      end
    end
  end

  def requirements
    if !self.is_valid_type($config.fs.type)
      ConfigError.new(
        ["config.fs.type"], # options concerned
        self.rm_prefix("\n - "), # suggest option
        'missing'
      )
    end

    # NFS checks
    nfs = $config.fs.type == 'nfs'
    if nfs 
      if Vagrant::Util::Platform.windows?
        raise "NFS won't going to work with windows hosts (try WSL)"
      end
      
      Vagrant::Util::Platform.linux? ? system('apt-get install nfs-kernel-server nfs-common') : nil

		  if Vagrant::Util::Platform.darwin? && !Vagrant.has_plugin?('vagrant-bindfs')
        system('vagrant plugin install vagrant-bindfs')
      end
    end
  end
# end Class Fs
end

# -*- mode: ruby -*-
# vi: set ft=ruby :

class Provider < Component
  PREFIX = 'pvd_'

  def initialize
    super(PREFIX)
    requirements
    # TODO dynamic function call and provider config < ENV var
    self.send(PREFIX+$config.provider.type)
  end

  # create virtualbox config with VboxManage settings
  def pvd_virtualbox
    $vagrant.vm.provider 'virtualbox' do |vb|
      $config.provider.opts.each do |name, param|
        # TODO if param is list loop it
        vb.customize ['modifyvm', :id, "--#{name}", param]
      end
    end
  end

  def pvd_libvirt
    raise "libvirt isn't supported for now"
  end

  def pvd_vm_ware
    raise "vm_ware isn't supported for now"
  end

  def pvd_parallels
    raise "parallels isn't supported for now"
  end

  def pvd_docker
    raise "docker isn't supported for now"
  end

  def requirements
    if !self.is_valid_type($config.provider.type)
      ConfigError.new(
        ["config.provider.type"], # options concerned
        self.rm_prefix("\n - "), # suggest for option
        'missing'
      )
    end
  end
# end Class Provider#
end

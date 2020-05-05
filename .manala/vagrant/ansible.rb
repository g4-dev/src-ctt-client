# -*- mode: ruby -*-
# vi: set ft=ruby :

class Ansible < Component
  PREFIX = 'ans_'
  
  def initialize
    super(PREFIX)
    if requirements
      # dispatch
      parse_config
      self.send(@PREFIX+$config.ansible.type)
    end
  end

  def ans_local()
    playbook_path = $config.ansible.path || "/tmp/"
    $vagrant.vm.provision 'file',
      source: $__dir__+$config.ansible.playbook, 
      destination: playbook_path
    # execute vagrant native provision  
    $vagrant.vm.provision 'ansible_local' do |ansible|
      ansible.provisioning_path = "#{playbook_path}"
      ansible.playbook = $config.ansible.playbook
      ansible.become = true
      ansible.inventory_path = $config.ansible.inventory
      ansible.extra_vars = $config.ansible.extra_vars
    end
  end

  def ans_classic
    # TODO : provision from local ansible exe
  end

  def ans_worker
    $vagrant.vm.provision :shell,
      run: './utils/playbook-worker.sh',
      args: "#{@git_url} #{$config.ansible.sub_playbook} #{$config.ansible.inventory}"
  end

  def parse_config
    if $config.ansible.playbook
      @git_url = [
        $config.git.provider, 
        $config.git.org,
        $config.ansible.playbook
      ].join('/')
    end
  end

  def requirements
    if $config.ansible.disabled || !$config.git.org || !$config.ansible.playbook
      return false
    end

    if !self.is_valid_type($config.ansible.type)
      ConfigError.new(
        ["config.ansible.type"], # options concerned
        self.rm_prefix("\n - "), # suggest for option
        'missing'
      )
    end
  end
  # end class Ansible
end

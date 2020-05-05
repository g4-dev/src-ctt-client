# -*- mode: ruby -*-
# vi: set ft=ruby :

class Network < Component
  SUFFIX = '_network'
  def initialize
    super(SUFFIX)
    requirements
    # dispatch
    self.send("#{$config.network.type}_network")
    redirect_ports
    $config.network.dns ? dns : nil
    @ssl ? ssl : nil
  end

  def public_network()
    $vagrant.vm.network :private_network, ip: $config.network.ip
    preferred_interfaces = ['eth0.*', 'eth\d.*', 'enp0s.*', 'enp\ds.*', 'en0.*', 'en\d.*']
    host_interfaces = %x( VBoxManage list bridgedifs | grep ^Name ).gsub(/Name:\s+/, '').split("\n")
    network_interface_to_use = preferred_interfaces.map{ |pi| 
      host_interfaces.find { |vm| vm =~ /#{Regexp.new(pi)}/ } 
    }.compact[0]
    
    $vagrant.vm.network :public_network, bridge: network_interface_to_use #, adapter: "1"
    routing
  end 

  def private_network()
    $vagrant.vm.network :private_network, ip: $config.network.ip
  end

  def dns
    $vagrant.landrush.enabled = true
    $vagrant.landrush.tld = $config.domain
  end

  def redirect_ports()
    $config.network.ports.each do |port|
      $vagrant.vm.network :forwarded_port, id: port.id || guid,
        guest: port.guest, 
        host: port.host,
        auto_correct: port.auto_correct || true,
        disabled: port.disabled || false
    end
  end

  def routing
    if Vagrant::Util::Platform.darwin? 
      @gateway = `route -n get default | grep 'gateway' | awk '{print $2}'`.delete("\n")
    elsif Vagrant::Util::Platform.linux? 
      # Not tested
      @gateway = `ip route show`[/default.*/][/\d+\.\d+\.\d+\.\d+/]
    end

    $vagrant.vm.provision :shell, 
      run: "always", 
      path: "#{__dir__}/utils/routing.py", 
      args: "#{@gateway}"
  end

  def ssl
    # Not tested
    cert = $config.network.ssl.cert
    path = $config.network.ssl.path
    local_path = "#{__dir__}/.cert"

    $vagrant.trigger.after :up do |t|
      t.run = { inline: 
        "scp -P 22 vagrant@#{$config.domain}:#{path}/#{cert} #{local_path}"
      }

      if Vagrant::Util::Platform.darwin? || Vagrant::Util::Platform.linux?
        t.run = { inline: 
          "sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain #{local_path}/#{cert}"
        }
      else
        t.run = { inline: "certutil -enterprise -f -v -AddStore 'Root' '#{local_path}/#{cert}'"}
      end
    end
  end

  def requirements
    if !self.is_valid_type($config.network.type, true)
      ConfigError.new(
        ["$config.network.type"], # options concerned
        self.rm_prefix("\n - "), # suggest for option
        'missing'
      )
    end

    if $config.network.dns && !Vagrant.has_plugin?('landrush')
			system('vagrant plugin install landrush')
    end
    
    if $config.network.ssl.path && $config.network.ssl.cert
      @ssl = true
    end
  end
 # end class 
end

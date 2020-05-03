# -*- mode: ruby -*-
# vi: set ft=ruby :

# ruby modules
require 'json'
# utils
require_relative "utils/scalars"
require_relative "utils/component"
# vagrant components
require_relative 'base'
require_relative 'network'
require_relative 'provider'
require_relative "fs"
require_relative "ansible"

class VagrantBootstrap
  # passing to each module vagrant object and part of the config hashmap
  def initialize(user_config, vagrant)
    $vagrant = vagrant # Vagrant object
    $config  = parse_config(user_config) # Full config.json
    Base.new
    Network.new
    Provider.new
    Ansible.new
    Fs.new
  end

  def parse_config(user_config)
    default = JSON.parse(File.read(__dir__+'/.default.json'))
    # TODO: config path as argument
    $config = JSON.parse(
      default.deep_merge(JSON.parse(user_config)).to_json,
      object_class: OpenStruct
    )
  end
  # end class VagrantBootstrap
end

# FIXES :
# Add common fixes depending on actuals vagrant issues
# =====================
# vagrant/vbox dhcp error
class VagrantPlugins::ProviderVirtualBox::Action::Network
  def dhcp_server_matches_config?(dhcp_server, config)
    true
  end
end

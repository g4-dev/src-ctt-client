# -*- mode: ruby -*-
# vi: set ft=ruby :

# Use this class as Mother class for any vagrant component
# Don't forget to add the super(<YOUR-PREFIX>) method in Child initializer
class Component
  def initialize(prefix)
    @PREFIX = prefix
    @Map = self.methods.grep(/#{@PREFIX}/).map(&:to_s)
  end

  def is_valid_type(type, is_suffix = false)
    type && @Map.include?(!is_suffix ? @PREFIX+type : type+@PREFIX)
  end

  def rm_prefix(glue = "\n")
    str = glue
    @Map.each { |el| str += el.to_s.sub(/#{@PREFIX}/, '') + glue }
    return str.sub(/#{glue}$/, '')
  end
end

# Simplify errors 
# Show error concerned config and suggest options
class ConfigError
  S = "\n"
  BASE_MSG = "[=== Error in config ===]"+S

  def initialize(_concerned = "", _suggests = nil, _type = 'standard')
		@suggests = _suggests || BASE_MSG
		@concerned = _concerned 
    puts self.send(_type)
    exit
  end

  def standard()
    @error = @suggests += @concerned
  end

	def missing()
    @error = BASE_MSG
    @concerned ? @error += "Concerned : #{self.concerned}"+S : nil
    @suggests ? @error += "Available Options : #{@suggests}"+S : nil
  end

  def concerned
    @concerned.is_a?(Array) ? @concerned.join(' '): @concerned
  end
end

# Ruby additionals
class ::Hash
  def deep_merge(second)
      merger = proc { |key, v1, v2| Hash === v1 && Hash === v2 ? v1.merge(v2, &merger) : v2 }
      self.merge(second, &merger)
  end
end
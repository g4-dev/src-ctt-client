# -*- mode: ruby -*-
# vi: set ft=ruby :

require 'digest'

def guid()
  SecureRandom.base64(8).gsub("/","_").gsub(/=+$/,"")
end

def sub_ip(string)
  (Digest::SHA256.hexdigest string)[0..1].to_i(16) % 250 + 2
end

def port(string)
  (Digest::SHA256.hexdigest string)[1..4].to_i(16)+1000
end

def rand_string(length=6)
	string = ""
	chars = ("A".."Z").to_a
	length.times do
		string << chars[rand(chars.length-1)]
	end
	string
end

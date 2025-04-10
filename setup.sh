#!/bin/bash

systemctl enable NetworkManager
systemctl start NetworkManager
# Set a local ip for Wired Connection 1:
nmcli con mod "Wired connection 1" ipv4.addresses "192.168.1.69/24 192.168.1.254"
nmcli con mod "Wired connection 1" ipv4.dns "8.8.8.8 8.8.4.4"
nmcli con mod "Wired connection 1" ipv4.method manual
nmcli con up "Wired connection 1"

# Allow root login via ssh
sed -i 's/^PermitRootLogin yes/#PermitRootLogin no/' /etc/ssh/sshd_config
systemctl enable sshd
systemctl start sshd
#Open ssh port 22
firewall-cmd --permanent --add-port=22/tcp
firewall-cmd --reload

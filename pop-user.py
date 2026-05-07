#!/bin/bash

# COSMIC System Manager v2026.1
# Handles User MGMT, Global A11y, and Auto-Login

if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root (use sudo)."
   exit 1
fi

GREETER_CONF="/etc/greetd/cosmic-greeter.toml"

show_menu() {
    echo "------------------------------------------"
    echo "   COSMIC SYSTEM MANAGEMENT UTILITY"
    echo "------------------------------------------"
    echo "1) List All Users"
    echo "2) Add New User"
    echo "3) Remove User"
    echo "4) Enable Screen Reader for ALL (Login Screen)"
    echo "5) Disable Screen Reader for Login Screen"
    echo "6) Set/Change Auto-Login User"
    echo "7) Disable Auto-Login"
    echo "q) Quit"
    echo "------------------------------------------"
}

while true; do
    show_menu
    read -p "Selection: " choice

    case $choice in
        1)
            echo "--- System Users (UID >= 1000) ---"
            awk -F' ' '{ if ($3 >= 1000 && $3 < 60000) print $1 }' /etc/passwd
            ;;
        2)
            read -p "Enter new username: " NEWUSER
            adduser "$NEWUSER"
            read -p "Make this user an Admin? (y/n): " IS_ADMIN
            if [[ "$IS_ADMIN" == "y" ]]; then
                usermod -aG sudo "$NEWUSER"
            fi
            ;;
        3)
            read -p "Enter username to REMOVE (and delete home dir): " DELUSER
            deluser --remove-home "$DELUSER"
            ;;
        4)
            echo "Enabling Orca for the cosmic-greeter..."
            # COSMIC uses cosmic-config backends. We target the greeter user.
            sudo -u cosmic-greeter dbus-launch gsettings set org.gnome.desktop.a11y.applications screen-reader-enabled true
            echo "Global Screen Reader enabled for the Login Screen."
            ;;
        5)
            sudo -u cosmic-greeter dbus-launch gsettings set org.gnome.desktop.a11y.applications screen-reader-enabled false
            echo "Global Screen Reader disabled for the Login Screen."
            ;;
        6)
            read -p "Enter username for Auto-Login: " AL_USER
            if id "$AL_USER" &>/dev/null; then
                cat <<EOF > "$GREETER_CONF"
[terminal]
vt = "1"
[default_session]
command = "cosmic-comp systemd-cat -t cosmic-greeter cosmic-greeter"
user = "cosmic-greeter"
[initial_session]
command = "cosmic-session"
user = "$AL_USER"
EOF
                echo "Auto-login set for $AL_USER."
            else
                echo "Error: User does not exist."
            fi
            ;;
        7)
            # Remove the initial_session block to disable auto-login
            sed -i '/\[initial_session\]/,+2d' "$GREETER_CONF"
            echo "Auto-login disabled."
            ;;
        q)
            exit 0
            ;;
        *)
            echo "Invalid selection."
            ;;
    esac
done

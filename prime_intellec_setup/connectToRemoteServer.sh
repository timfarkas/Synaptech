#!/bin/bash

# Function to load profile
load_profile() {
    if [ -f "$1.profile" ]; then
        source "$1.profile"
    else
        echo "Profile $1.profile not found."
    fi
}

# Ask if the user wants to use the last profile
if [ -f last_profile.txt ]; then
    LAST_PROFILE=$(cat last_profile.txt)
    echo "The last profile used was: $LAST_PROFILE"
    read -p "Do you want to use the last profile? (y/n): " USE_LAST_PROFILE
    if [[ "$USE_LAST_PROFILE" == "y" || "$USE_LAST_PROFILE" == "Y" ]]; then
        PROFILE=$(cat last_profile.txt)
        load_profile "$PROFILE"
    else
        read -p "Do you want to use an alternative profile? (y/n): " USE_ALTERNATIVE_PROFILE
        if [[ "$USE_ALTERNATIVE_PROFILE" == "y" || "$USE_ALTERNATIVE_PROFILE" == "Y" ]]; then
            read -p "Enter the name of the alternative profile (e.g. 'V100'): " PROFILE
            load_profile "$PROFILE"
        else
            echo "Proceeding without loading any profile."
            PROFILE=""  # Ensure PROFILE is empty if no profile is selected
        fi
    fi
else
    echo "No last profile found. Proceeding without loading any profile."
    PROFILE=""  # Ensure PROFILE is empty if no profile is found
fi


# Use the information from the last profile if it was loaded
if [ -n "$PROFILE" ]; then
    SERVER_NAME="$PROFILE"
    SERVER_IP="${SERVER_IP:-$SERVER_IP}"
    USERNAME="${USERNAME:-root}"
    PORT="${PORT:-18326}"
    PEM_FILE="${PEM_FILE:-private_key.pem}"
else
    # Ask for the server name only if profile was not loaded
    read -p "Enter the server name: " SERVER_NAME

    # Ask for the IP address of the remote server, default to last used if available
    read -p "Enter the IP address of the remote server (default: ${SERVER_IP:-}): " SERVER_IP
    SERVER_IP=${SERVER_IP:-$SERVER_IP}

    # Ask for the username (defaults to root)
    read -p "Enter the SSH username (default: ${USERNAME:-root}): " USERNAME
    USERNAME=${USERNAME:-${USERNAME:-root}}

    # Ask for the SSH port, default to last used if available
    read -p "Enter the SSH port (default: ${PORT:-18326}): " PORT
    PORT=${PORT:-${PORT:-18326}}

    # Ask for the name of the .pem SSH key file
    read -p "Enter the name of your .pem SSH key file (default: ${PEM_FILE:-private_key.pem}): " PEM_FILE
    PEM_FILE=${PEM_FILE:-${PEM_FILE:-private_key.pem}}
fi


ssh "$USERNAME"@"$SERVER_IP" -p "$PORT" -i "$PEM_FILE"

# Save the current profile name for future use
echo "$SERVER_NAME" > last_profile.txt
EOF
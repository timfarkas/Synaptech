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
    PROJECT_NAME="${PROJECT_NAME:-MicroSynBrand2022}"
    GIT_URL=${GIT_URL:-https://github.com/timfarkas/BrainTransform.git}
    IS_PRIVATE="${IS_PRIVATE:-Y}"
    GIT_PAT=${GIT_PAT:-$(grep 'GITHUB_PAT' .env | cut -d '=' -f2)}
    echo "Profile loaded successfully"
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

    # Ask for the name of the project
    read -p "Enter the name of the project (default: ${PROJECT_NAME:-MicroSynBrand2022}): " PROJECT_NAME
    PROJECT_NAME=${PROJECT_NAME:-MicroSynBrand2022}

    # Ask for the URL of the Git repository
    if [ -z "$GIT_URL" ]; then
        read -p "Enter the URL of the Git repository (default: ${GIT_URL:-https://github.com/timfarkas/BrainTransform.git}): " GIT_URL
        GIT_URL=${GIT_URL:-https://github.com/timfarkas/BrainTransform.git}
    fi

    if [ -z "$IS_PRIVATE" ]; then
        # Ask if the repository is private
        read -p "Is the repository private? (y/n): " IS_PRIVATE
    fi

    # If the repository is private, ask for a Personal Access Token (PAT)
    if [[ "$IS_PRIVATE" == "y" || "$IS_PRIVATE" == "Y" ]]; then
        if [ -z "$GIT_PAT" ]; then
            read -sp "Enter your Personal Access Token (PAT): " GIT_PAT
            GIT_PAT=${GIT_PAT:-$(grep 'GITHUB_PAT' .env | cut -d '=' -f2)}
            echo
        fi
    fi
fi

# Ask for the desired mode
read -p "Enter the desired mode (default, validation, or profiling): " MODE
MODE=${MODE:-default}
MODE=${MODE,,}


# Save the current configuration to a profile file
echo "SERVER_IP=$SERVER_IP" > "$SERVER_NAME.profile"
echo "USERNAME=$USERNAME" >> "$SERVER_NAME.profile"
echo "PORT=$PORT" >> "$SERVER_NAME.profile"
echo "PEM_FILE=$PEM_FILE" >> "$SERVER_NAME.profile"
echo "PROJECT_NAME=$PROJECT_NAME" >> "$SERVER_NAME.profile"
echo "GIT_URL=$GIT_URL" >> "$SERVER_NAME.profile"
echo "IS_PRIVATE=$IS_PRIVATE" >> "$SERVER_NAME.profile"
if [[ "$IS_PRIVATE" == "y" || "$IS_PRIVATE" == "Y" ]]; then
    echo "GIT_PAT=$GIT_PAT" >> "$SERVER_NAME.profile"
else
    echo "GIT_PAT=" >> "$SERVER_NAME.profile"
fi


# Save the current profile name for future use
echo "$SERVER_NAME" > last_profile.txt

if [[ "$IS_PRIVATE" == "y" || "$IS_PRIVATE" == "Y" ]]; then
    GIT_URL=$(echo "$GIT_URL" | sed -e "s|https://|https://$GIT_PAT@|")
fi


# SSH into the remote server and initialize the project
ssh -i "$PEM_FILE" "$USERNAME"@"$SERVER_IP" -p "$PORT" << EOF
# Determine if 'sudo' is available and if the user is root
if [ "\$(id -u)" -eq 0 ]; then
    # Running as root, no need for 'sudo'
    SUDO=""
elif command -v sudo >/dev/null 2>&1; then
    # 'sudo' is available
    SUDO="sudo"
else
    # 'sudo' is not available, and not running as root
    echo "This script requires root privileges to install packages."
    echo "However, 'sudo' is not available, and you are not running as root."
    echo "Please run this script as root or install 'sudo'."
    exit 1
fi && \
cd ../workspace || { echo "Failed to change directory to ../workspace. Proceeding in the current directory."; } && \
mkdir -p "$PROJECT_NAME" && cd "$PROJECT_NAME" || { echo "mkdir: cannot create directory '$PROJECT_NAME': No such file or directory"; exit 1; } && \
if [ ! -d ".git" ]; then git init -b main; fi && \
if git remote | grep -q origin; then git remote set-url origin "$GIT_URL"; else git remote add origin "$GIT_URL"; fi && \
git config user.email "mail@timfarkas.com" && \
git config user.name "Tim Farkas" && \
git fetch || { echo "Failed to fetch from remote repository. Please check the URL and your network connection."; exit 1; } && \
if ! git show-ref --verify --quiet refs/heads/main; then \
    git checkout -b main origin/main; \
else \
    git branch --set-upstream-to=origin/main main; \
    git pull origin main; \
fi && \
echo "Project '$PROJECT_NAME' has been set up with the Git repository '$GIT_URL'." && \
echo "Installing dependencies" && \
if ! command -v pip3 > /dev/null 2>&1; then \
    echo "pip3 is required but it's not installed. Installing pip3..."; \
    \$SUDO apt-get update && \$SUDO apt-get install -y python3-pip; \
fi && \
if ! command -v python3 > /dev/null 2>&1; then \
    echo "python3 is required but it's not installed. Installing python3..."; \
    \$SUDO apt-get update && \$SUDO apt-get install -y python3; \
fi && \
if ! command -v virtualenv > /dev/null 2>&1; then \
    echo "virtualenv is required but it's not installed. Installing virtualenv..."; \
    pip install --upgrade virtualenv; \
fi && \
virtualenv venv && \
source venv/bin/activate && \
pip3 install -r requirements.txt && \
if ! command -v unzip > /dev/null 2>&1; then \
    echo "unzip is required but it's not installed. Installing unzip..."; \
    \$SUDO apt-get update && \$SUDO apt-get install -y unzip; \
fi && \
if ! command -v unzip > /dev/null 2>&1; then \
    echo "Failed to install unzip."; \
    exit 1; \
else \
    echo "unzip is installed."; \
fi
echo "Loading datasets via nohup."  && \
nohup ./loadDataSets.sh -c -m "$MODE" > loadDataSets.sh.log 2>&1 && \
echo "Success initializing repository and initializing data set loading." && \
exec "$SHELL"
EOF
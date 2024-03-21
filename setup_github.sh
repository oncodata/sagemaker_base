#!/bin/bash

# Function to display messages in green color
print_green() {
    printf "\033[0;32m%s\033[0m\n" "$1"
}

# Function to display messages in yellow color
print_yellow() {
    printf "\033[0;33m%s\033[0m\n" "$1"
}

# Function to display messages in red color
print_red() {
    printf "\033[0;31m%s\033[0m\n" "$1"
}

# Step 1: Generate SSH Key Pair
print_yellow "Step 1: Generating SSH Key Pair"
read -p "Enter your email address (used for generating SSH key): " email

ssh-keygen -t rsa -b 4096 -C "$email"
print_green "SSH key pair generated successfully!"

# Step 2: Add SSH Key to GitHub
print_yellow "Step 2: Adding SSH Key to GitHub"
cat ~/.ssh/id_rsa.pub
print_yellow "Copy the above SSH key and add it to your GitHub account."
read -p "Press [Enter] after adding the SSH key to your GitHub account..."

# Step 3: Test SSH Connection
print_yellow "Step 3: Testing SSH Connection to GitHub"
ssh -T git@github.com
print_green "SSH connection to GitHub successful!"

# Step 4: Clone Repository
print_yellow "Step 4: Cloning Repository"
read -p "Enter the GitHub repository URL (SSH format): " repo_url
git clone $repo_url
print_green "Repository cloned successfully!"
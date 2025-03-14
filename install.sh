#!/bin/bash

# Step 1: Create the directory and set permissions
mkdir -p /run/user/0
chown root:root /run/user/0

# Step 2: Install system dependencies
apt-get update
apt-get install -y flac

# Step 3: Install Python dependencies
pip install -r requirements.txt

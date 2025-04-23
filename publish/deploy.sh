#!/bin/bash

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Create app directory
mkdir -p /opt/app
cd /opt/app

# Copy application files
# Note: You would typically get these from a Git repository or S3 bucket
# For example:
# git clone <your-repo> .
# or
# aws s3 cp s3://your-bucket/app.tar.gz .
# tar xzf app.tar.gz

# Build and start containers
sudo docker-compose up -d --build

# Set up Nginx as reverse proxy (optional)
sudo apt-get install -y nginx
sudo cat > /etc/nginx/sites-available/app << EOL
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
EOL

sudo ln -s /etc/nginx/sites-available/app /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Set up SSL with Let's Encrypt (optional)
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com 
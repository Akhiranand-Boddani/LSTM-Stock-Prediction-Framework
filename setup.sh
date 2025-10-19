#!/bin/bash

# LSTM Stock Prediction - Automated Setup Script
# This script automates the initial setup and GitHub upload process

echo "================================================"
echo "LSTM Stock Prediction - Setup & GitHub Upload"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# Check if git is installed
print_info "Checking if git is installed..."
if ! command -v git &> /dev/null; then
    print_error "Git is not installed. Please install git first."
    echo "Visit: https://git-scm.com/downloads"
    exit 1
fi
print_success "Git is installed"

# Check if Python is installed
print_info "Checking if Python is installed..."
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    print_error "Python is not installed. Please install Python 3.8+."
    exit 1
fi
print_success "Python is installed"

echo ""
echo "================================================"
echo "Step 1: Git Initialization"
echo "================================================"

# Initialize git repository
if [ -d .git ]; then
    print_info "Git repository already initialized"
else
    print_info "Initializing git repository..."
    git init
    print_success "Git repository initialized"
fi

echo ""
echo "================================================"
echo "Step 2: Configure Git User"
echo "================================================"

# Check if git user is configured
GIT_NAME=$(git config user.name)
GIT_EMAIL=$(git config user.email)

if [ -z "$GIT_NAME" ]; then
    read -p "Enter your name: " USER_NAME
    git config user.name "$USER_NAME"
    print_success "Git user name set to: $USER_NAME"
else
    print_info "Git user name: $GIT_NAME"
fi

if [ -z "$GIT_EMAIL" ]; then
    read -p "Enter your email: " USER_EMAIL
    git config user.email "$USER_EMAIL"
    print_success "Git email set to: $USER_EMAIL"
else
    print_info "Git email: $GIT_EMAIL"
fi

echo ""
echo "================================================"
echo "Step 3: Stage Files"
echo "================================================"

print_info "Staging all files..."
git add .
print_success "Files staged"

# Show what will be committed
echo ""
print_info "Files to be committed:"
git status --short

echo ""
read -p "Do you want to continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_error "Setup cancelled"
    exit 1
fi

echo ""
echo "================================================"
echo "Step 4: Create Initial Commit"
echo "================================================"

print_info "Creating initial commit..."
git commit -m "Initial commit: LSTM Stock Prediction Framework"
print_success "Initial commit created"

echo ""
echo "================================================"
echo "Step 5: GitHub Repository Setup"
echo "================================================"

echo ""
print_info "Please create a repository on GitHub:"
echo "  1. Go to https://github.com/new"
echo "  2. Repository name: lstm-stock-prediction"
echo "  3. Description: LSTM-based stock price prediction with Streamlit"
echo "  4. Visibility: Public (recommended)"
echo "  5. DO NOT initialize with README, .gitignore, or license"
echo "  6. Click 'Create repository'"
echo ""

read -p "Enter your GitHub username: " GITHUB_USERNAME

if [ -z "$GITHUB_USERNAME" ]; then
    print_error "GitHub username is required"
    exit 1
fi

REPO_URL="https://github.com/${GITHUB_USERNAME}/lstm-stock-prediction.git"

echo ""
print_info "Repository URL: $REPO_URL"

# Check if remote already exists
if git remote | grep -q "^origin$"; then
    print_info "Remote 'origin' already exists. Updating URL..."
    git remote set-url origin "$REPO_URL"
else
    print_info "Adding remote repository..."
    git remote add origin "$REPO_URL"
fi

print_success "Remote repository configured"

echo ""
echo "================================================"
echo "Step 6: Push to GitHub"
echo "================================================"

print_info "Renaming branch to 'main'..."
git branch -M main
print_success "Branch renamed to 'main'"

echo ""
print_info "Pushing to GitHub..."
echo ""
print_info "Note: You may be prompted for GitHub credentials"
print_info "If using 2FA, use a Personal Access Token instead of password"
print_info "Generate token at: https://github.com/settings/tokens"
echo ""

if git push -u origin main; then
    print_success "Successfully pushed to GitHub!"
    echo ""
    echo "================================================"
    echo "✓ Setup Complete!"
    echo "================================================"
    echo ""
    print_success "Your repository is now available at:"
    echo "  https://github.com/${GITHUB_USERNAME}/lstm-stock-prediction"
    echo ""
    print_info "Next steps:"
    echo "  1. Visit your repository on GitHub"
    echo "  2. Deploy to Streamlit Cloud: https://share.streamlit.io"
    echo "  3. See DEPLOYMENT.md for detailed instructions"
    echo ""
else
    print_error "Failed to push to GitHub"
    echo ""
    print_info "Common issues and solutions:"
    echo "  1. Authentication failed:"
    echo "     → Use Personal Access Token instead of password"
    echo "     → Generate at: https://github.com/settings/tokens"
    echo ""
    echo "  2. Repository doesn't exist:"
    echo "     → Make sure you created the repository on GitHub first"
    echo "     → Check the repository name matches: lstm-stock-prediction"
    echo ""
    echo "  3. Permission denied:"
    echo "     → Verify your GitHub username is correct"
    echo "     → Check repository visibility settings"
    echo ""
    print_info "Manual push command:"
    echo "  git push -u origin main"
    exit 1
fi

echo ""
read -p "Would you like to open your repository in the browser? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    REPO_WEB_URL="https://github.com/${GITHUB_USERNAME}/lstm-stock-prediction"
    
    # Detect OS and open browser
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        open "$REPO_WEB_URL"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        xdg-open "$REPO_WEB_URL" 2>/dev/null || print_info "Please open: $REPO_WEB_URL"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        # Windows (Git Bash)
        start "$REPO_WEB_URL"
    else
        print_info "Please open: $REPO_WEB_URL"
    fi
fi

echo ""
print_success "Setup script completed successfully!"
echo ""
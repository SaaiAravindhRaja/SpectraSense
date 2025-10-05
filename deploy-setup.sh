#!/bin/bash

echo "🚀 SpectraSense Deployment Setup"
echo "================================"
echo ""

# Check if git is initialized
if [ ! -d .git ]; then
    echo "📦 Initializing git repository..."
    git init
    echo "✅ Git initialized"
else
    echo "✅ Git already initialized"
fi

# Check git status
echo ""
echo "📊 Current git status:"
git status --short

# Add all files
echo ""
echo "📝 Adding files to git..."
git add .
echo "✅ Files staged"

# Show what will be committed
echo ""
echo "📋 Files to be committed:"
git diff --cached --name-only

# Prompt for commit
echo ""
read -p "💬 Commit message (press Enter for default): " commit_msg
if [ -z "$commit_msg" ]; then
    commit_msg="Deploy SpectraSense AI to production"
fi

git commit -m "$commit_msg"
echo "✅ Changes committed"

# Check if remote exists
echo ""
if git remote | grep -q origin; then
    echo "✅ Remote 'origin' already configured"
    echo "📍 Remote URL: $(git remote get-url origin)"
else
    echo "⚠️  No remote configured yet"
    echo ""
    echo "Next steps:"
    echo "1. Create a new repository on GitHub: https://github.com/new"
    echo "2. Name it: SpectraSense"
    echo "3. Run these commands:"
    echo ""
    echo "   git remote add origin https://github.com/YOUR_USERNAME/SpectraSense.git"
    echo "   git branch -M main"
    echo "   git push -u origin main"
fi

echo ""
echo "✨ Setup complete!"
echo ""
echo "📚 Next: Read DEPLOYMENT.md for deployment instructions"

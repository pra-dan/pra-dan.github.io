#!/bin/sh

# If a command fails then the deploy stops
set -e

printf "\033[0;32mDeploying updates to GitHub...\033[0m\n"

# Build the project.
hugo -t hugo-future-imperfect-slim # if using a theme, replace with `hugo -t <YOURTHEME>`

# Go To Public folder
cd public

# Add changes to git.
git add *

# Commit changes.
msg="rebuilding site $(date)"
if [ -n "$*" ]; then
	msg="$*"
fi
git commit -m "$msg"

git remote add pra-dan.github.io https://github.com/pra-dan/pra-dan.github.io.git 

# Push source and build repos.
git push -u pra-dan.github.io -f origin master

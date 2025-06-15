#!/bin/bash

COMMIT_MSG=${1:-"Updated changes"}

git checkout develop || exit
git pull origin develop || exit

git add .
git commit -m "$COMMIT_MSG"
git push origin develop || exit

git checkout main || exit
git pull origin main || exit

git merge develop || exit
git add .
git commit -m "Merged develop into main"
git push origin main

git checkout develop


# GitHub Workflows

This directory contains GitHub Actions workflow configurations for continuous integration (CI) of the ContextGem project.


## Available Workflows

### tests (`ci-tests.yml`)

**Features:**
- Runs on multiple operating systems (Ubuntu, macOS, Windows)
- Tests across Python versions 3.10, 3.11, 3.12, and 3.13
- Checks formatting with Black
- Runs test suite with VCR (recorded API responses)
- Generates test coverage reports

**Trigger:**
- Automatically runs on push and pull request events on the main branch
- Can be triggered manually through the GitHub Actions UI

**Environment Variables:**
- This workflow uses the following environment variables:
    - `CONTEXTGEM_OPENAI_API_KEY`: Secret OpenAI API key
    - `GIST_SECRET`: Secret token to upload coverage results to a gist for badge generation

### CodeQL Analysis (`codeql.yml`)

This workflow performs code security scanning using GitHub's CodeQL analysis engine.

**Features:**
- Scans Python codebase for security vulnerabilities and coding errors
- Analyzes code quality and identifies potential issues
- Results are available in the Security tab of the repository

**Trigger:**
- Automatically runs on push and pull request events on the main and dev branches
- Scheduled to run weekly
- Can be triggered manually through the GitHub Actions UI

### Documentation Build (`docs.yml`)

This workflow builds and deploys the project documentation to GitHub Pages.

**Features:**
- Builds documentation using Sphinx
- Deploys documentation to GitHub Pages when merged to main
- Creates preview builds on pull requests

**Trigger:**
- Automatically runs on push and pull request events on the main branch
- Can be triggered manually through the GitHub Actions UI

### Check Contributor Agreement (`contributor-agreement-check.yml`)

This workflow ensures all contributors have signed the Contributor Agreement by checking for properly filled agreement files.

**Features:**
- Verifies that each contributor has a signed agreement file
- Ensures agreement files are not empty and contain the contributor's username
- Prevents deletion of existing contributor agreement files
- Posts helpful comments on PRs when agreement requirements aren't met

**Trigger:**
- Automatically runs on all pull request events (opened, synchronized, reopened)


## Running Workflows

- **tests:** These run automatically on push/PR to the main branch
- **CodeQL Analysis:** Runs automatically on push/PR to main/dev, weekly, and manually
- **Documentation Build:** Runs automatically on push/PR to main and manually
- **Check Contributor Agreement:** Runs automatically on all PRs

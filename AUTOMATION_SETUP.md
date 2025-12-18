# Automated PyPI and ReadTheDocs Setup

This document explains how to set up automatic publishing to PyPI and ReadTheDocs using **Trusted Publishing** (no API tokens required!).

## 🤖 Automated Workflows

### 1. **Publish to PyPI** (Production)
- **Trigger**: When you push a git tag starting with `v` (e.g., `v1.0.0`)
- **File**: `.github/workflows/publish.yml`
- **What it does**: Automatically builds and publishes to PyPI using Trusted Publishing

### 2. **ReadTheDocs**
- **Trigger**: Automatic on every push to `main`
- **File**: `.readthedocs.yaml`
- **What it does**: Automatically builds documentation

## 📋 Setup Instructions

### Step 1: Set Up PyPI Trusted Publishing (Recommended - No Tokens!)

Trusted Publishing is PyPI's secure, token-free publishing method.

1. **Go to PyPI**:
   - Visit https://pypi.org and log in
   - Go to "Your projects" (you'll create the project after first upload)

2. **Add GitHub as Trusted Publisher** (do this BEFORE first upload):
   - Go to https://pypi.org/manage/account/publishing/
   - Click "Add a new pending publisher"
   - Fill in:
     - **PyPI Project Name**: `polerisk`
     - **Owner**: `kylejones200`
     - **Repository name**: `PoleRisk`
     - **Workflow name**: `publish.yml`
     - **Environment name**: `pypi`
   - Click "Add"

3. **That's it!** No API tokens needed.

### Step 2: Set Up ReadTheDocs

1. Go to [readthedocs.org](https://readthedocs.org) and sign in with GitHub
2. Click **Import a Project**
3. Select `kylejones200/PoleRisk` from your repositories
4. Click **Next**
5. Configure:
   - Name: `polerisk`
   - Repository URL: `https://github.com/kylejones200/PoleRisk`
   - Default branch: `main`
6. Click **Finish**

ReadTheDocs will automatically:
- Build docs on every push to `main`
- Build docs for pull requests
- Host your docs at `https://polerisk.readthedocs.io/`

## 🚀 How to Use

### Publishing a New Version to PyPI:

```bash
# 1. Update version in pyproject.toml and polerisk/__init__.py
# Example: version = "1.0.1"

# 2. Commit changes
git add pyproject.toml polerisk/__init__.py
git commit -m "Bump version to 1.0.1"
git push origin main

# 3. Create and push a tag
git tag v1.0.1
git push origin v1.0.1

# 4. GitHub Actions automatically publishes to PyPI! ✨
```

**That's it!** The workflow will:
- Build the package
- Publish to PyPI using Trusted Publishing
- No tokens or secrets needed

### Creating a GitHub Release (Optional):

After pushing the tag, you can also create a GitHub release for better visibility:

1. Go to https://github.com/kylejones200/PoleRisk/releases/new
2. Select your tag: `v1.0.1`
3. Release title: `Release 1.0.1`
4. Add release notes describing changes
5. Click **Publish release**

### Documentation Updates:

Documentation automatically updates when you push to `main`:

```bash
# Just push to main - ReadTheDocs auto-rebuilds!
git push origin main
```

## 🔄 Complete Workflow Example

Here's a complete release workflow:

```bash
# 1. Make your changes
git checkout -b feature/new-feature
# ... make changes ...
git commit -m "Add new feature"
git push origin feature/new-feature

# 2. Create PR and merge to main
# (via GitHub UI)

# 3. Update version for release
git checkout main
git pull origin main

# Edit pyproject.toml: version = "1.0.1"
# Edit polerisk/__init__.py: __version__ = "1.0.1"

git add pyproject.toml polerisk/__init__.py
git commit -m "Bump version to 1.0.1"
git push origin main

# 4. Tag and release
git tag v1.0.1
git push origin v1.0.1

# ✨ Automatic deployment starts!
```

## ✅ Verification

After setup, verify everything works:

1. **Test PyPI Publishing**:
   - Create a test tag: `git tag v1.0.0 && git push origin v1.0.0`
   - Check GitHub Actions: https://github.com/kylejones200/PoleRisk/actions
   - Verify package on PyPI: https://pypi.org/project/polerisk/

2. **Test ReadTheDocs**:
   - Push a doc change to `main`
   - Check ReadTheDocs builds: https://readthedocs.org/projects/polerisk/builds/
   - Verify docs are live: https://polerisk.readthedocs.io/

## 🛠️ Troubleshooting

### PyPI Publishing Fails

**Error: Trusted publishing exchange failure**
- Verify you added the trusted publisher on PyPI
- Check all fields match exactly:
  - Repository: `kylejones200/PoleRisk`
  - Workflow: `publish.yml`
  - Environment: `pypi`

**Error: Project name already taken**
- The package name `polerisk` might be taken
- Try a different name in `pyproject.toml`

**Error: Version already exists**
- You're trying to upload a version that already exists
- Increment the version number in `pyproject.toml`

### ReadTheDocs Build Fails

**Error: Python version**
- Check `.readthedocs.yaml` has Python 3.13
- Verify requirements-docs.txt has all dependencies

**Error: Sphinx configuration**
- Check `docs/source/conf.py` is valid
- Ensure all Sphinx extensions are installed

**Error: Import errors**
- Add missing dependencies to `requirements-docs.txt`
- Ensure package installs correctly

## 🔐 Security Benefits of Trusted Publishing

Trusted Publishing is more secure than API tokens because:
- ✅ No long-lived credentials to manage
- ✅ No secrets stored in GitHub
- ✅ Uses OpenID Connect (OIDC) for authentication
- ✅ Scoped to specific repository and workflow
- ✅ PyPI's recommended method

## 📞 Support

- **GitHub Actions**: Check the Actions tab for build logs
- **PyPI Trusted Publishing**: https://docs.pypi.org/trusted-publishers/
- **ReadTheDocs**: https://docs.readthedocs.io/

## 🎉 You're All Set!

Your workflow is now:

1. **Develop** → Make changes and merge to main
2. **Release** → Push a version tag (v1.0.0) → Auto-publish to PyPI
3. **Document** → Push to main → Auto-update ReadTheDocs

No manual building, uploading, or token management needed! 🚀

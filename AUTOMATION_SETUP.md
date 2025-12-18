# Automated PyPI and ReadTheDocs Setup

This document explains how to set up automatic publishing to PyPI and ReadTheDocs.

## 🤖 Automated Workflows

Your repository now has two GitHub Actions workflows:

### 1. **Publish to PyPI** (Production)
- **Trigger**: When you create a GitHub Release
- **File**: `.github/workflows/publish-pypi.yml`
- **What it does**: Automatically builds and publishes to PyPI

### 2. **Publish to TestPyPI** (Testing)
- **Trigger**: When you push to `develop` or `staging` branches
- **File**: `.github/workflows/publish-test-pypi.yml`
- **What it does**: Automatically builds and publishes to TestPyPI for testing

## 📋 Setup Instructions

### Step 1: Get PyPI API Tokens

#### For Production PyPI:
1. Go to [pypi.org](https://pypi.org) and log in
2. Go to Account Settings → API tokens
3. Click "Add API token"
4. Name it: `GitHub Actions - polerisk`
5. Scope: Select "Entire account" (or specific to `polerisk` after first upload)
6. Copy the token (starts with `pypi-`)

#### For TestPyPI (Optional but Recommended):
1. Go to [test.pypi.org](https://test.pypi.org) and log in
2. Go to Account Settings → API tokens
3. Click "Add API token"
4. Name it: `GitHub Actions - polerisk-test`
5. Scope: "Entire account"
6. Copy the token (starts with `pypi-`)

### Step 2: Add Secrets to GitHub

1. Go to your GitHub repository: https://github.com/kylejones200/polerisk
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add these secrets:

   **Secret 1:**
   - Name: `PYPI_API_TOKEN`
   - Value: Your PyPI token (from Step 1)
   
   **Secret 2 (Optional):**
   - Name: `TEST_PYPI_API_TOKEN`
   - Value: Your TestPyPI token (from Step 1)

### Step 3: Set Up ReadTheDocs

1. Go to [readthedocs.org](https://readthedocs.org) and sign in with GitHub
2. Click **Import a Project**
3. Select `kylejones200/polerisk` from your repositories
4. Click **Next**
5. Configure:
   - Name: `polerisk`
   - Repository URL: `https://github.com/kylejones200/polerisk`
   - Default branch: `main`
6. Click **Finish**

ReadTheDocs will automatically:
- Build docs on every push to `main`
- Build docs for pull requests
- Host your docs at `https://polerisk.readthedocs.io/`

## 🚀 How to Use

### Publishing to PyPI (Production)

1. **Update version** in `pyproject.toml` and `polerisk/__init__.py`:
   ```toml
   version = "1.0.1"  # Increment version
   ```

2. **Commit and push** changes:
   ```bash
   git add pyproject.toml polerisk/__init__.py
   git commit -m "Bump version to 1.0.1"
   git push origin main
   ```

3. **Create a GitHub Release**:
   - Go to https://github.com/kylejones200/polerisk/releases
   - Click **Draft a new release**
   - Tag version: `v1.0.1` (must start with 'v')
   - Release title: `Release 1.0.1`
   - Description: List changes/improvements
   - Click **Publish release**

4. **Automatic publishing**:
   - GitHub Actions will automatically build and publish to PyPI
   - Check progress in the **Actions** tab
   - Package will be live at `https://pypi.org/project/polerisk/`

### Testing with TestPyPI

1. **Create a `develop` branch**:
   ```bash
   git checkout -b develop
   ```

2. **Make changes and push**:
   ```bash
   git add .
   git commit -m "Test changes"
   git push origin develop
   ```

3. **Automatic publishing**:
   - GitHub Actions will automatically publish to TestPyPI
   - Test installation:
     ```bash
     pip install --index-url https://test.pypi.org/simple/ polerisk
     ```

### Manual Publishing (If Needed)

You can also manually trigger the workflows:

1. Go to **Actions** tab in GitHub
2. Select the workflow (Publish to PyPI or TestPyPI)
3. Click **Run workflow**
4. Select the branch
5. Click **Run workflow**

## 📚 ReadTheDocs Automatic Updates

ReadTheDocs automatically rebuilds documentation when:
- You push to the `main` branch
- You create a pull request (preview build)
- You create a new tag/release

**View your docs at**: https://polerisk.readthedocs.io/

To manually rebuild:
1. Go to https://readthedocs.org/projects/polerisk/
2. Click **Builds**
3. Click **Build Version**

## 🔄 Workflow Summary

### Development Workflow:
```
1. Make changes → 2. Push to develop → 3. Auto-publish to TestPyPI → 4. Test
```

### Release Workflow:
```
1. Update version → 2. Commit & push → 3. Create GitHub Release → 4. Auto-publish to PyPI
```

### Documentation Workflow:
```
1. Update docs → 2. Push to main → 3. ReadTheDocs auto-builds → 4. Live at readthedocs.io
```

## ✅ Verification

After setup, verify everything works:

1. **Test PyPI Publishing**:
   - Push to `develop` branch
   - Check Actions tab for successful build
   - Verify package on TestPyPI

2. **Test PyPI Production**:
   - Create a test release (v1.0.0)
   - Check Actions tab for successful build
   - Verify package on PyPI: https://pypi.org/project/polerisk/

3. **Test ReadTheDocs**:
   - Push a doc change to `main`
   - Check ReadTheDocs builds: https://readthedocs.org/projects/polerisk/builds/
   - Verify docs are live: https://polerisk.readthedocs.io/

## 🛠️ Troubleshooting

### PyPI Publishing Fails

**Error: Invalid token**
- Verify the token is correct in GitHub Secrets
- Ensure token hasn't expired
- Check token scope includes the package

**Error: File already exists**
- You're trying to upload a version that already exists
- Increment the version number in `pyproject.toml`

**Error: Package name taken**
- The package name `polerisk` might be taken
- Try a different name in `pyproject.toml`

### ReadTheDocs Build Fails

**Error: Python version**
- Check `.readthedocs.yaml` has Python 3.12
- Verify requirements-docs.txt has all dependencies

**Error: Sphinx configuration**
- Check `docs/source/conf.py` is valid
- Ensure all Sphinx extensions are installed

**Error: Import errors**
- Add missing dependencies to `requirements-docs.txt`
- Ensure package installs correctly

## 📞 Support

- **GitHub Actions**: Check the Actions tab for build logs
- **PyPI Issues**: https://pypi.org/help/
- **ReadTheDocs Issues**: https://docs.readthedocs.io/

## 🎉 You're All Set!

Once configured, your workflow is:

1. **Develop** → Push to `develop` → Auto-test on TestPyPI
2. **Release** → Create GitHub Release → Auto-publish to PyPI
3. **Document** → Push to `main` → Auto-update ReadTheDocs

No manual building or uploading needed! 🚀


# How to Publish `setiastrosuitepro` to PyPI

## Prerequisites
- You need an account on [PyPI](https://pypi.org/).
- You need an API Token from PyPI (Account Settings -> API Tokens).

## Steps

1. **Verify Build** (Optional but recommended)
   Run this command to make sure everything builds correctly:
   ```powershell
   poetry build
   ```

2. **Configure Token (One-time setup)**
   Run this command, replacing `pypi-XXXXXXXX` with your actual token:
   ```powershell
   poetry config pypi-token.pypi pypi-XXXXXXXX
   ```

3. **Publish**
   Run the publish command:
   ```powershell
   poetry publish
   ```
   
   *Note: `poetry publish --build` combines the build and publish steps.*

## Troubleshooting
- If you get version conflict errors, make sure to bump the version in `pyproject.toml` (e.g. `poetry version patch`) before publishing again.

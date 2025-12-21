# Line Ending Configuration

This repository uses Git's automatic line ending normalization to support contributors on both Windows and Unix-like systems (Linux/Mac).

## How It Works

- **Repository**: Files are stored with LF (`\n`) line endings for consistency
- **Working Directory**: Files are automatically converted to your platform's native line endings:
  - **Windows**: CRLF (`\r\n`) in your working directory
  - **Linux/Mac**: LF (`\n`) in your working directory
- **Git handles conversion automatically** on checkout and commit

## Setup Instructions

### Windows Users
```bash
git config --global core.autocrlf true
```
This tells Git to:
- Convert LF → CRLF when checking out files
- Convert CRLF → LF when committing files

### Linux/Mac Users
```bash
git config --global core.autocrlf input
```
This tells Git to:
- Keep LF as-is when checking out files
- Convert CRLF → LF when committing files (in case someone committed with CRLF)

## One-Time Normalization (After Setup)

After setting up `.gitattributes` and configuring `core.autocrlf`, normalize existing files:

```bash
git add --renormalize .
git commit -m "Normalize line endings"
```

## Benefits

✅ **No conflicts**: Everyone works with their platform's native line endings  
✅ **Consistent repository**: All files stored as LF in Git  
✅ **Automatic conversion**: Git handles everything transparently  
✅ **Clean diffs**: Only actual code changes show up, not line ending differences  

## Troubleshooting

If you see line ending changes in your diffs:

1. Verify your `core.autocrlf` setting:
   ```bash
   git config core.autocrlf
   ```

2. Re-normalize your working directory:
   ```bash
   git add --renormalize .
   ```

3. Check `.gitattributes` is committed:
   ```bash
   git ls-files .gitattributes
   ```

## Files That Keep Specific Endings

Some files are explicitly configured:
- `*.sh`, `*.bash`: Always LF (required for Unix scripts)
- `*.bat`, `*.cmd`, `*.ps1`: Always CRLF (Windows scripts)

All other text files use automatic normalization based on your platform.


# Development Environment Troubleshooting

## Local Wheel Installation Issues

### Problem: `import core_logging` fails after `uv sync`

**Symptoms:**
- `uv sync` completes successfully
- `python -c "import core_logging"` fails with `ModuleNotFoundError`
- sck-core-framework appears to be installed but modules aren't accessible

**Root Cause:**
uv's handling of local wheel files through `tool.uv.sources` is unreliable, especially on Windows with complex dependency graphs.

### Solution: Hybrid Approach

**Step 1: Build the dependency wheel**
```bash
cd ../sck-core-framework
uv build  # Creates dist/sck_core_framework-*.whl
```

**Step 2: Manual virtual environment setup**
```bash
cd ../sck-core-ai
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
# OR
source .venv/bin/activate     # Linux/Mac
```

**Step 3: Install with pip (not uv)**
```bash
# Install the local wheel first
pip install ../sck-core-framework/dist/sck_core_framework-*.whl

# Then install the current project
pip install -e .

# Add dev dependencies
pip install pytest black flake8 mypy
```

**Step 4: Verify**
```bash
python -c "import core_logging; print('Success!')"
```

## Why This Works

### Traditional Python Tooling (Reliable)
- `python -m venv`: Creates predictable, isolated environments
- `pip install wheel.whl`: Mature wheel installation logic
- Well-tested with local packages and editable installs

### uv Limitations (Current)
- Local wheel path resolution can fail
- `tool.uv.sources` implementation is still evolving
- Complex dependency resolution sometimes fails where pip succeeds
- Virtual environment management can interfere with local packages

## Best Practices

### For Development
1. **Use hybrid approach**: Manual venv + pip for local wheels + uv for PyPI packages
2. **Build dependencies first**: Always build required wheels before installing dependents
3. **Verify imports**: Test critical imports after each setup step
4. **Document paths**: Keep track of relative paths between projects

### For Production
- uv is excellent for production deployments with PyPI packages
- Consider building and uploading internal wheels to private PyPI for production use

### When to Use What

| Use Case | Tool | Reason |
|----------|------|---------|
| Local wheel installation | pip | Mature, reliable |
| PyPI package management | uv | Fast, modern |
| Virtual environment creation | `python -m venv` | Predictable |
| Production deployments | uv | Fast, reproducible |
| CI/CD pipelines | uv | Excellent for PyPI deps |

## Common Mistakes to Avoid

1. **Wrong virtual environment**: Always verify `(project-name)` in prompt
2. **Cross-project contamination**: Never run commands from wrong project directory
3. **Assuming uv handles everything**: Use the right tool for each job
4. **Skipping verification**: Always test imports after installation
5. **Opening new terminals**: Use existing terminals to maintain environment context

## Future Improvements

As uv matures, the local wheel handling will likely improve. Until then, this hybrid approach provides the reliability of traditional Python tooling with the speed benefits of uv where it excels.
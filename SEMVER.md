# Semantic Versioning (SemVer)

## Overview

Semantic Versioning is a versioning scheme that provides a standardized way to communicate the nature of changes in software releases. It helps developers and users understand the impact of updates and make informed decisions about when to upgrade.

## Version Format

Semantic versions follow the format: **MAJOR.MINOR.PATCH**

```
X.Y.Z
│ │ └── Patch version (bug fixes, no breaking changes)
│ └──── Minor version (new features, backward compatible)
└────── Major version (breaking changes, major updates)
```

### Examples
- `1.0.0` - Initial release
- `1.2.3` - Major version 1, Minor version 2, Patch version 3
- `2.0.0` - Major version bump (breaking changes)
- `1.3.0` - Minor version bump (new features)
- `1.2.4` - Patch version bump (bug fixes)

## Version Components

### MAJOR Version (X)
- **Breaking changes** that are not backward compatible
- Complete API redesigns
- Removal of deprecated features
- Major architectural changes
- **Increment when:** Breaking changes are introduced

### MINOR Version (Y)
- **New functionality** added in a backward compatible manner
- New features that don't break existing functionality
- Deprecation notices (features marked for future removal)
- **Increment when:** New features are added

### PATCH Version (Z)
- **Bug fixes** that are backward compatible
- Performance improvements
- Documentation updates
- Minor bug fixes
- **Increment when:** Bug fixes are implemented

## Pre-release and Build Metadata

### Pre-release Versions
```
X.Y.Z-alpha.1    # Alpha release
X.Y.Z-beta.2     # Beta release
X.Y.Z-rc.3       # Release candidate
```

### Build Metadata
```
X.Y.Z+build.123  # Build information
X.Y.Z+sha.abc123 # Git commit hash
```

## When to Increment Versions

### Increment MAJOR (X)
- [ ] Breaking changes to public APIs
- [ ] Incompatible changes to existing functionality
- [ ] Major architectural changes
- [ ] Removal of deprecated features

### Increment MINOR (Y)
- [ ] New features added
- [ ] Deprecation notices added
- [ ] Significant improvements to existing features
- [ ] New optional dependencies

### Increment PATCH (Z)
- [ ] Bug fixes
- [ ] Performance improvements
- [ ] Documentation updates
- [ ] Minor code refactoring

## Best Practices

### 1. Start with 0.x.x
- Use `0.x.x` for initial development
- Indicates the software is not yet stable
- Breaking changes are expected

### 2. Release 1.0.0 When Ready
- Release `1.0.0` when the API is stable
- Commit to maintaining backward compatibility
- Document the public API clearly

### 3. Maintain Backward Compatibility
- Avoid breaking changes in patch and minor releases
- Use deprecation warnings before removing features
- Provide migration guides for major version updates

### 4. Document Changes
- Keep a detailed changelog
- Document breaking changes clearly
- Provide migration guides for major updates

### 5. Use Semantic Versioning Consistently
- Apply the same rules across all components
- Use automated tools to enforce versioning
- Tag releases in version control

## 🚀 Practical Development Workflow

### Branching Strategy

```
main (production)
├── develop (integration)
├── feature/feature-name
├── bugfix/bug-description
└── hotfix/urgent-fix
└── release/v1.2.0
└── chore/update-dependencies
```

**Branch Naming Convention:**
- `feature/` - New features
- `bugfix/` - Bug fixes
- `hotfix/` - Urgent production fixes
- `release/` - Release preparation
- `chore/` - Maintenance tasks

### Commit Message Format

Use **Conventional Commits** format for all commit messages:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat:` - New feature (increments MINOR)
- `fix:` - Bug fix (increments PATCH)
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

**Examples:**
```bash
# Feature commit
git commit -m "feat: add user authentication system"

# Bug fix commit
git commit -m "fix: resolve memory leak in data processor"

# Documentation commit
git commit -m "docs: update API documentation"

# Breaking change (increments MAJOR)
git commit -m "feat!: remove deprecated login endpoint"

# Bug fix commit
git commit -m "chore: update dependencies"

# Test commit
git commit -m "test: add unit tests for new feature"

# Refactor commit
git commit -m "refactor: improve code readability"

```

### Daily Development Workflow

1. **Start Work:**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes:**
   - Write code following project standards
   - Write/update tests
   - Update documentation if needed

3. **Commit Changes:**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

4. **Push and Create PR:**
   ```bash
   git push origin feature/your-feature-name
   # Create Pull Request on GitHub/GitLab
   ```

5. **Code Review:**
   - Address review comments
   - Update PR as needed
   - Squash commits if requested

6. **Merge:**
   - Merge to `develop` branch
   - Delete feature branch

### Release Process

#### Preparing a Release

1. **Create Release Branch:**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b release/v1.2.0
   ```

2. **Update Version:**
   ```bash
   # Update version in setup.py, pyproject.toml, or __init__.py
   # Update CHANGELOG.md
   git add .
   git commit -m "chore: prepare release v1.2.0"
   ```

3. **Test Release:**
   - Run full test suite
   - Test installation
   - Verify documentation

4. **Merge to Main:**
   ```bash
   git checkout main
   git merge release/v1.2.0
   git tag v1.2.0
   git push origin main
   git push origin v1.2.0
   ```

5. **Merge Back to Develop:**
   ```bash
   git checkout develop
   git merge main
   git push origin develop
   ```

6. **Cleanup:**
   ```bash
   git branch -d release/v1.2.0
   git push origin --delete release/v1.2.0
   ```

#### Hotfix Process

1. **Create Hotfix Branch:**
   ```bash
   git checkout main
   git checkout -b hotfix/critical-bug-fix
   ```

2. **Fix the Issue:**
   - Make minimal changes to fix the bug
   - Update tests
   - Update version (PATCH increment)

3. **Test and Commit:**
   ```bash
   git add .
   git commit -m "fix: resolve critical security vulnerability"
   ```

4. **Release:**
   ```bash
   git checkout main
   git merge hotfix/critical-bug-fix
   git tag v1.2.1
   git push origin main
   git push origin v1.2.1
   ```

5. **Update Develop:**
   ```bash
   git checkout develop
   git merge main
   git push origin develop
   ```

## Changelog Format

```
# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [1.2.0] - 2024-01-15
### Added
- New feature A
- New feature B

### Changed
- Improved performance of feature X

### Deprecated
- Feature Y is deprecated and will be removed in 2.0.0

### Fixed
- Bug in feature Z

## [1.1.0] - 2024-01-01
### Added
- Initial feature set

### Fixed
- Various bug fixes
```

## Tools and Automation

### Version Management
- **Python**: `bumpversion`, `semantic-release`
- **Node.js**: `semantic-release`, `standard-version`
- **Git**: Git tags for releases

### CI/CD Integration
- Automate version bumping
- Generate changelogs
- Create release notes
- Tag releases automatically

## Common Mistakes to Avoid

### ❌ Don't
- Skip version numbers (1.0.0 → 1.2.0)
- Use non-standard version formats
- Ignore breaking changes in minor releases
- Forget to update changelogs

### ✅ Do
- Increment versions consistently
- Document all changes
- Test backward compatibility
- Communicate breaking changes clearly

## Quick Reference Commands

### Version Management
```bash
# Check current version
git describe --tags --abbrev=0

# List all tags
git tag -l

# Create annotated tag
git tag -a v1.2.0 -m "Release version 1.2.0"

# Push tags
git push origin --tags
```

### Branch Management
```bash
# List all branches
git branch -a

# Delete local branch
git branch -d branch-name

# Delete remote branch
git push origin --delete branch-name
```

### Commit History
```bash
# View commit history
git log --oneline

# View commits since last tag
git log $(git describe --tags --abbrev=0)..HEAD --oneline

# View commits in a branch
git log main..feature-branch --oneline
```

## Resources

- [Semantic Versioning Specification](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
- [Conventional Commits](https://www.conventionalcommits.org/)

## Example Workflow

1. **Development**: Work on features in development branch
2. **Feature Complete**: Merge to main, increment MINOR version
3. **Bug Fix**: Fix bug, increment PATCH version
4. **Breaking Change**: Plan breaking change, increment MAJOR version
5. **Release**: Tag release, update changelog, deploy

---

*This document follows the Semantic Versioning 2.0.0 specification.*

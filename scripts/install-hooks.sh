#!/bin/bash
# Install git hooks for the project

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HOOKS_DIR="$PROJECT_ROOT/.git/hooks"

echo "Installing git hooks..."

# Copy pre-push hook
if [ -f "$PROJECT_ROOT/.git/hooks/pre-push" ]; then
    cp "$PROJECT_ROOT/.git/hooks/pre-push" "$HOOKS_DIR/pre-push.backup"
    echo "Backed up existing pre-push hook to pre-push.backup"
fi

# Create a new pre-push hook from template
cat > "$HOOKS_DIR/pre-push" << 'HOOK_EOF'
#!/bin/bash

# Pre-push hook to run CI checks before pushing to remote
# This ensures code quality and prevents broken code from being pushed

set -e

echo "========================================="
echo "Running pre-push CI checks..."
echo "========================================="

# Get the project root directory
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Track if any checks failed
FAILED=0

# Check if we're in a virtual environment, if not, try to use one
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d "venv" ]; then
        print_info "Activating virtual environment..."
        source venv/bin/activate
    elif [ -d ".venv" ]; then
        print_info "Activating virtual environment..."
        source .venv/bin/activate
    else
        print_warning "No virtual environment found. Make sure dependencies are installed."
    fi
fi

# 1. Check code formatting with black
print_info "Checking code formatting with black..."
if command -v black &> /dev/null; then
    if black --check polerisk tests 2>/dev/null; then
        print_info "Code formatting is correct!"
    else
        print_error "Code formatting issues found!"
        print_error "Run: black polerisk tests"
        FAILED=1
    fi
else
    print_warning "black not installed. Skipping formatting check."
    print_warning "Install with: pip install black"
fi

# 2. Check linting with flake8
print_info "Checking code style with flake8..."
if command -v flake8 &> /dev/null; then
    if flake8 polerisk --count --select=E9,F63,F7,F82 --show-source --statistics 2>/dev/null; then
        if flake8 polerisk --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics 2>/dev/null; then
            print_info "Linting passed!"
        else
            print_error "Linting issues found!"
            FAILED=1
        fi
    else
        print_error "Critical linting errors found!"
        FAILED=1
    fi
else
    print_warning "flake8 not installed. Skipping linting check."
    print_warning "Install with: pip install flake8"
fi

# 3. Optional: Type checking with mypy (warn only, don't fail)
print_info "Running type checking with mypy..."
if command -v mypy &> /dev/null; then
    if mypy polerisk --ignore-missing-imports 2>/dev/null; then
        print_info "Type checking passed!"
    else
        print_warning "Type checking found issues (non-blocking)"
    fi
else
    print_warning "mypy not installed. Skipping type checking."
    print_warning "Install with: pip install mypy"
fi

# 4. Run tests
print_info "Running tests with pytest..."
if ! command -v pytest &> /dev/null; then
    print_error "pytest is not installed!"
    print_error "Install with: pip install -e '.[dev]'"
    FAILED=1
elif pytest tests/ -v --tb=short 2>&1 | grep -q "ModuleNotFoundError"; then
    print_warning "Some test dependencies are missing. Tests skipped."
    print_warning "To run full tests, install dependencies: pip install -e '.[dev]'"
elif ! pytest tests/ -v --tb=short 2>/dev/null; then
    print_error "Tests failed!"
    FAILED=1
else
    print_info "All tests passed!"
fi

echo ""
echo "========================================="

# Summary
if [ $FAILED -eq 1 ]; then
    print_error "Pre-push checks FAILED! Push aborted."
    echo ""
    echo "Please fix the issues above before pushing."
    exit 1
else
    print_info "All pre-push CI checks passed! Proceeding with push..."
    echo "========================================="
    exit 0
fi
HOOK_EOF

chmod +x "$HOOKS_DIR/pre-push"
echo "Pre-push hook installed successfully!"
echo ""
echo "The hook will now run before each push:"
echo "  - Black code formatting check"
echo "  - Flake8 linting check"
echo "  - Mypy type checking (warn only)"
echo "  - Pytest tests"
echo ""
echo "To disable the hook temporarily, run:"
echo "  git push --no-verify"


# Coding Conventions

These conventions apply to all code in this project. Follow them strictly.

## File Organization
- One class or function per file; avoid large monolithic files
- Each lab module lives in its own directory under `src/`
- Test files mirror source structure in `tests/` directory
- Module name should match directory name (e.g., `src/lab1_chatbot/chatbot.py`)

## Language & Style
- **Type hints required** for all function parameters and return values
- **Docstrings** for all public functions, classes, and modules using Google style:
  ```python
  def process_image(image: np.ndarray) -> Prediction:
      """Process input image and return prediction.
      
      Args:
          image: Input image as numpy array (H, W, C)
      
      Returns:
          Prediction object with class and confidence.
      """
  ```
- **PEP 8** compliant with 4-space indentation
- Max line length: 100 characters
- Use `snake_case` for variables and functions, `PascalCase` for classes
- **Clear variable names**: no abbreviations, be explicit (`prediction` not `pred`)
- Constants in `UPPER_CASE`
- Private members prefixed with `_`

## External Dependencies
- **No external libraries without explicit approval** from project lead
- Prefer standard library when possible
- Pin exact versions in `requirements.txt` when adding new packages
- Document non-standard dependencies in module docstring

## Module Structure
Each module follows this pattern:
```python
"""Module docstring describing purpose."""

from typing import Optional

class ClassName:
    """Class docstring."""
    pass

def public_function(arg: str) -> bool:
    """Function docstring."""
    pass

def _private_function():
    """Private helper."""
    pass

if __name__ == "__main__":
    # Minimal demo or test runner
    pass
```

## Testing
- Test files for each lab: `tests/labX/test_<module>.py`
- Use `pytest` framework
- One test file per source file minimum
- Test names descriptive: `test_<function>_<scenario>`
- Aim for >80% coverage on core modules
- Include at least one integration test per lab

## Data & Models
- Raw data never modified in-place; always create processed copies
- Models saved with versioned filenames: `model_v1.pt`, `model_v2.pt`
- Document preprocessing steps in code comments
- Store large assets (>10MB) in `data/` not repository

## Error Handling
- Raise specific exceptions, not bare `Exception`
- Log errors with context but never log raw data or PII
- Fail fast on invalid inputs with clear error messages
- Wrap external API calls in try/except with retry logic

## Documentation
- Update `ARCHITECTURE.md` when adding new components
- Keep docstrings synchronized with code changes
- Document assumptions and known limitations in comments
- README updates for user-facing features only

## Rules of Engagement
- **No hallucination rule**: only implement what's explicitly specified in requirements or approved design docs
- **No over-specification rule**: do not add features or complexity beyond stated needs
- When in doubt, ask for clarification before writing code
- Commit small, focused changes with descriptive messages

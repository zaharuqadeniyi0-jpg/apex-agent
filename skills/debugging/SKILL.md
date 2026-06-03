name: debugging
description: "Systematic debugging approach — understand, isolate, fix, verify."
category: software-development
version: "1.0.0"

## Debugging Skill

When debugging, follow this 4-phase approach:

### Phase 1: Understand
- Read the error message carefully
- Identify the file, line number, and error type
- Read the relevant source code
- Check recent changes (git log, git diff)

### Phase 2: Isolate
- Create a minimal reproduction
- Add print/logging statements
- Check assumptions (types, values, paths)
- Use the appropriate debugger (pdb, debugpy, node --inspect)

### Phase 3: Fix
- Apply the minimal fix
- Don't refactor while fixing
- Write a test that would have caught the bug

### Phase 4: Verify
- Run the test suite
- Confirm the original error is gone
- Check for regressions

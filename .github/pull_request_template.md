## Description
<!-- Provide a concise summary of the changes and motivation. -->

## Related Issue
<!-- Link the issue here, e.g., Closes #123 -->

## Type of Change
- [ ] `feat`: New feature or engine capability
- [ ] `fix`: Bug fix
- [ ] `tree`: Domain knowledge tree addition or modification
- [ ] `docs`: Documentation update
- [ ] `refactor`: Code refactoring without behavior change
- [ ] `test`: New or updated tests

## Verification & Testing
<!-- Describe the tests and verification commands you ran to verify your changes. -->
- [ ] `python scripts/lint_trees.py` passed cleanly
- [ ] `python scripts/check_static.py` passed cleanly
- [ ] `python nstl_cli.py compile` and `validate` passed (100% success rate)
- [ ] Automated tests passed (`pytest tests/...`)

## Checklist
- [ ] My branch is rebased/merged with the latest `dev` branch
- [ ] My code follows the project's conventions and does not introduce hardcoded constants
- [ ] I have updated documentation if applicable
- [ ] No temporary files, diff patches, or local databases are committed

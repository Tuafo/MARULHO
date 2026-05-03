# Coding Standards

<!-- Customize this file with your project's coding standards.
     The reviewer agent loads it during code review via @.sandcastle/CODING_STANDARDS.md
     so these standards are enforced during review without costing tokens during implementation. -->

## Style

- Use snake_case for variables and functions
- Use PascalCase for classes, types and exceptions
- Use UPPER_SNAKE_CASE for constants
- Prefer named exports over default exports
- Prefer having in file docs

## Testing

- Every public function must have at least one test
- Use descriptive test names that explain the expected behavior

## Architecture

- Keep modules focused on a single responsibility
- Prefer composition over inheritance
# Issue tracker — GitHub

Issues are tracked in **GitHub** at **Tuafo/MARULHO**.

## Commands

| Action | Command |
|--------|---------|
| Create issue | `gh issue create` |
| List issues | `gh issue list` |
| View issue | `gh issue view <number>` |
| Close issue | `gh issue close <number>` |
| Add labels | `gh issue edit <number> --add-label <label>` |
| Remove labels | `gh issue edit <number> --remove-label <label>` |

## Conventions

- Always use the `gh` CLI — never scrape the web UI.
- When creating an issue, include a clear title and body; assign relevant triage labels.
- Reference issues by number (e.g. `#42`) in commits and PRs.

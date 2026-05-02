# Triage labels

These are the label strings used by the `triage` skill when moving issues through the state machine.

| Role | Label | Meaning |
|------|-------|---------|
| needs-triage | `needs-triage` | Maintainer needs to evaluate |
| needs-info | `needs-info` | Waiting on reporter for more details |
| ready-for-agent | `ready-for-agent` | Fully specified, AFK-ready |
| ready-for-human | `ready-for-human` | Needs human implementation |
| wontfix | `wontfix` | Will not be actioned |

## Notes

- These labels must exist in the GitHub repo. Create them before first use if they don't already exist.
- If you rename a label in GitHub, update this file to match.

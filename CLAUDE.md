# Claude Code Project Configuration

## Repository Setup
- **Primary work location**: Fork at `Marazzano/morphseq`
- **Main branch**: `diffusion-dev` (not `main`)
- **Workflow**: Work on fork, PR to own `diffusion-dev` branch

## Git Configuration
- Fork remote: `fork` → `git@github.com:Marazzano/morphseq.git`
- When creating PRs: use `--base diffusion-dev`
- Example PR creation: `gh pr create --base diffusion-dev --title "Feature title"`

## Project Structure
- **Main codebase**: `/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/`
- **Annotation system**: `scripts/annotations/`
- **Tests**: `scripts/tests/`
- **Utilities**: `scripts/utils/`
- **Pipelines**: `scripts/pipelines/`

## Development Workflow
- Always work on feature branches off `diffusion-dev`
- Push to `fork` remote, not `origin`
- Test critical validation logic changes
- Follow existing code patterns and conventions

## Recent Major Changes
- **Embryo Annotation System Refactored** (2024): Fixed monkey patching, inheritance violations, and added comprehensive type hints
- **Configuration Management**: Moved hardcoded validation lists to `scripts/annotations/config.json`

## Environment
- **Conda Environment**: `segmentation_grounded_sam`
- **Working Directory**: `/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox`

## Important Notes
- This fork-based workflow keeps all development contained within the user's repository
- The `diffusion-dev` branch serves as the main development branch
- All Claude configuration files are gitignored to prevent accidental commits
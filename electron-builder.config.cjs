const fs = require('fs')
const path = require('path')
const base = require('./package.json').build

const candidates = [process.env.MAH_RUNTIME_ROOT, path.join(__dirname, 'models-at-home'), path.resolve(__dirname, '..', 'models-at-home')]
const runtime = candidates.find(p => p && fs.existsSync(path.join(p, 'homellm')))

module.exports = {
  ...base,
  beforePack() {
    if (!runtime) throw new Error('ML runtime missing: set MAH_RUNTIME_ROOT before packaging')
  },
  extraResources: [
    { from: 'examples/world-local', to: 'examples/world-local' },
    { from: 'requirements-world-local.txt', to: 'requirements-world-local.txt' },
    { from: 'requirements-world-foundation.txt', to: 'requirements-world-foundation.txt' },
    { from: 'docs', to: 'docs', filter:['*.md','*.json'] },
    { from: 'README.md', to: 'README.md' },
    { from: 'CHANGELOG.md', to: 'CHANGELOG.md' },
    { from: 'RELEASE.md', to: 'RELEASE.md' },
    { from: 'examples/world-models', to: 'examples/world-models' },
    { from: 'examples/vla', to: 'examples/vla' },
    { from: 'backend', to: 'backend', filter: ['**/*', '!**/__pycache__/**', '!**/*.pyc', '!tests/**'] },
    {
      from: runtime || 'models-at-home', to: 'models-at-home',
      filter: ['homellm/**/*','configs/**/*','blueprints/**/*','study_materials/**/*','scripts/**/*','src/**/*',
        'Dockerfile','requirements.txt','pyproject.toml','LICENSE','README*.md','ARCHITECTURE.md','docker-compose.yml',
        '!**/__pycache__/**', '!**/*.pyc', '!**/*.egg-info/**', '!**/.env', '!**/.env.*', '!**/unsloth_compiled_cache/**',
        '!models/**', '!datasets/**', '!out/**', '!logs/**', '!.runs/**', '!.cache/**',
        '!.git/**', '!.venv/**', '!venv/**', '!**/.ipynb_checkpoints/**'],
    },
  ],
}

# Glacier Ecosystem Agent Guidelines & Rules

## 🚨 MANDATORY SECURITY RULE: ZERO SECRETS POLICY & PRE-PUBLISH AUDITING

### 1. ABSOLUTE PROHIBITION ON EMBEDDING SECRETS
- **NEVER** write, commit, or publish API keys, access tokens, passwords, database connection strings, private keys, or credentials in any file.
- **NEVER** store real keys in `appsettings.json`, `appsettings.*.json`, `.env`, `.yaml`, `.xml`, `.cs`, `.py`, `.json`, or script files.
- All secrets and credentials must strictly be loaded from **Environment Variables** (e.g. `Environment.GetEnvironmentVariable("GEMINI_API_KEY")`) or local secret managers (`dotnet user-secrets`).
- `appsettings*.json`, `*.env*`, and credential files must ALWAYS be listed in `.gitignore`. Only sanitized templates (e.g., `appsettings.template.json`) with empty values may be committed.

### 2. MANDATORY PRE-PUBLISH & PRE-COMMIT SECURITY AUDIT
Before executing ANY of the following actions:
- `git commit`
- `git push`
- `gh repo create`
- `dotnet nuget push`
- Publishing or creating any public or private repository, release, or package

The agent **MUST** run the automated security scanner:
```bash
python scripts/security_check.py <target_directory>
```
If the scanner detects ANY potential secret, or returns a non-zero exit code:
- **HALT IMMEDIATELY**.
- Do NOT proceed with the commit, push, or publish.
- Remove and sanitize the secret and re-verify clean status before proceeding.

### 3. AUTOMATIC REMEDIATION PROTOCOL
If any credential or key is ever detected as exposed:
1. Immediately switch the remote repository to **PRIVATE**.
2. Immediately notify the user to **REVOKE and ROTATE** the key at the provider console.
3. Purge the secret from all files and rewrite/amend git history.
4. Force-push the purged clean history.

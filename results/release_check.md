# Release check of the full history

Checked on 2026-10-10, before the repository was first pushed. The scan covers **every commit reachable from any ref**
(35 commits up to `3fda024`; the commit that adds this file adds nothing else) and **every blob in them** (2601 blobs, 80.2 MB,
of which 1183 binary: `.npz`, `.png`, `.gif`). The history was produced from the development repository with `git filter-repo`
(internal working notes removed, author e-mail replaced by the GitHub noreply address, internal labels taken out of commit messages).

## Summary

| Check | Result |
|---|---|
| Secrets / tokens / passwords (patterns) | **0** |
| Secret-related keywords | 10 hits, all harmless (`HF_TOKEN` in a Hugging Face warning text and a doc note that it is *not* set; `TOKENIZERS_PARALLELISM=false`) |
| High-entropy strings (≥ 32 chars, entropy ≥ 4.5 bits/char, non-hex) | 12 hits, all harmless (GitHub commit URLs, a `/tmp` path) |
| Long hex strings | 268 × 40 hex (git commit hashes in `results/*/config.json` `git_commit`, one LeRobot commit URL), 35 × 64 hex (SHA-256 checksums of checkpoints / configs, one SHA-256 test vector); no keys |
| Blobs > 5 MB | **0** (largest: 1.93 MB, `results/t5_checks/n4_delay_te/traj/batch_0000.npz`) |
| E-mail addresses in files | **0** |
| Author / committer identities | 1: `YaoYinJie <171040871+2181385109@users.noreply.github.com>` (70 = 35 commits × author + committer) |
| Personal-mail-provider addresses (files and commit metadata) | **0** |
| Phone numbers (CN mobile, +86) | **0** |
| ID-number-like strings (18-digit with a valid birth date) | **0** |
| School / university names | **0** |
| Local user paths (`C:\Users\…`, `/home/…`, Windows user name) | **0** (the only hits of the user-name digits are inside floating-point numbers in `episodes.csv`) |
| Removed files in any commit | **0** (by path and by object id, see below) |
| Internal task labels in commit messages | **0** |

## Commands and results

**1. Removed files never appear** (`CLAUDE.md`, `HANDOFF.md`, `docs/notes_zh/`, `.pytest_cache/`):

```bash
for c in $(git rev-list --all); do git ls-tree -r --name-only $c; done \
  | grep -c -E '^(CLAUDE\.md|HANDOFF\.md|docs/notes_zh/|"docs/notes_zh/|\.pytest_cache/)'   # -> 0
git rev-list --objects --all | grep -c -E 'CLAUDE\.md|HANDOFF\.md|notes_zh|pytest_cache'     # -> 0
git log --all --format= --name-only | grep -c -E 'CLAUDE\.md|HANDOFF\.md|notes_zh|pytest_cache'  # -> 0
```

Object check against the development repository (read-only): the 71 object ids (blobs and trees) that these paths had anywhere in the
development history were listed with `git -C <dev-repo> rev-list --objects --all`, and `git cat-file -e <id>` was run for each in this
repository: **0 of 71 present**. `git fsck --unreachable --no-reflogs` reports 0 unreachable objects, and there is no `refs/original`.

**2. Large blobs:**

```bash
git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
  | awk '$1=="blob" && $3 > 5*1024*1024'        # -> no output
```

**3. Identities and commit messages:**

```bash
git log --all --format='%an <%ae>%n%cn <%ce>' | sort | uniq -c
# -> 70 YaoYinJie <171040871+2181385109@users.noreply.github.com>
git log --all --format=%B | grep -c -E '\bT([1-9]|1[0-3])[ab]?\b'               # -> 0
git log --all --format=%B | grep -i 'co-authored-by'
# -> Co-Authored-By: Claude <noreply@anthropic.com>   (one commit, kept unchanged)
```

Commit messages were also searched for internal-process wording (the Chinese terms for coordinator, prompt and executor session,
"run by user", the names of the removed note files): 0 hits.

**4. Content scan of every blob** (a Python script reading each blob with `git cat-file blob`; text blobs fully, binary blobs only for
key-block / token markers and mail addresses). Patterns:

- secrets: `-----BEGIN … PRIVATE KEY-----`, AWS (`AKIA…`/`ASIA…`), GitHub (`ghp_`/`gho_`/`ghu_`/`ghs_`/`ghr_`, `github_pat_`),
  Hugging Face (`hf_` + 30 chars), `sk-…`, Slack (`xox?-`), Google (`AIza…`), JWT, `key/secret/token/password = value` assignments,
  URLs with embedded credentials;
- keywords: password, passwd, secret, token, api key, credential, private key;
- high entropy: runs of ≥ 32 characters from `[A-Za-z0-9+/_=-]`, Shannon entropy ≥ 4.5 bits per character, hex-only runs counted separately;
- personal information: e-mail addresses, the personal mail provider of the original author address, CN mobile numbers
  (11 digits starting 13–19, not inside a longer number or a decimal), `+86` numbers, 18-character ID numbers with a valid birth date,
  school words (university, college, school, student id, and their Chinese equivalents), `C:\Users\`, `/home/`, messenger names.

Results are in the summary table. Every keyword and high-entropy hit was read; none is a credential.

## Notes

- The files under `results/` are raw outputs and are unchanged. Their logs and `config.json` files contain absolute paths of the
  machine they were produced on (WSL paths such as `/root/...` and the development checkout path) and `git_commit` hashes of the development history.
- The real name of the author appears only as the git author name, as intended.

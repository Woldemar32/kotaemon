`ktem` provides user management when enabled in `flowsettings.py` (default in this fork: `KH_FEATURE_USER_MANAGEMENT = True`).

Set in `flowsettings.py` or via environment (`decouple` / `.env` where supported):

- `KH_FEATURE_USER_MANAGEMENT` — enable login and per-user data.
- `KH_FEATURE_USER_MANAGEMENT_ADMIN` — admin username created on first start.
- `KH_FEATURE_USER_MANAGEMENT_PASSWORD` — admin password for that user.

When enabled:

- **Welcome** tab — sign in / sign out (other tabs hidden until login).
- **Settings** — change password.
- **Resources → Users** — create, list, edit, delete users (**admin only**; non-admin users do not see the Resources tab).

In **SSO mode** (`KH_SSO_ENABLED`), the Resources tab is hidden; manage models via `flowsettings.py` / `.env` instead.

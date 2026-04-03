# Get In Touch – Email Setup

Messages from the **Get In Touch** form are sent to **info.evolvora@gmail.com**.

## One-time setup (Gmail App Password)

1. Use the Gmail account **info.evolvora@gmail.com** (or the account you want to send from).
2. Enable **2-Step Verification** for that account (Google Account → Security).
3. Create an **App Password**:
   - Go to [Google Account → Security → App passwords](https://myaccount.google.com/apppasswords).
   - Create a new app password (e.g. name: "SignifyConnect").
   - Copy the 16-character password.

## Save password so you don’t type it every time (recommended)

1. In your project folder, copy the example file and rename it to `.env`:
   - Copy `.env.example` → `.env`  
   - Or create a new file named `.env` in the same folder as `app.py`.

2. Open `.env` in Notepad and put your app password on one line (no quotes needed):
   ```
   GMAIL_APP_PASSWORD=abcd efgh ijkl mnop
   ```
   Replace `abcd efgh ijkl mnop` with your real 16-character app password. Save and close.

3. Run the app as usual. You do **not** need to set the password in CMD anymore:
   ```cmd
   cd "c:\Users\hriti\OneDrive\Desktop\Sign language detection"
   python app.py
   ```

The app reads `GMAIL_APP_PASSWORD` from `.env` automatically. **Do not share or commit `.env`** (it is listed in `.gitignore`).

---

### Alternative: set password in CMD each time

If you prefer not to use a `.env` file:

**PowerShell:**
```powershell
$env:GMAIL_APP_PASSWORD = "your-16-char-app-password"
python app.py
```

**Command Prompt:**
```cmd
set GMAIL_APP_PASSWORD=your-16-char-app-password
python app.py
```

If `GMAIL_APP_PASSWORD` is not set, the contact form will show: *"Contact form is not configured."*

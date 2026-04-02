# Telegram Context Bot (Daily Summaries + Q&A)

This bot:
1. Stores incoming messages (text + captions; plus basic media metadata) from chats.
2. Posts a once-a-day summary to each chat at `19:00`.
3. Answers `/ask <question>` using the latest daily summary + the most relevant recent messages.

## Quickstart

1. Install dependencies:
   ```bash
   cd telegram-context-bot
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Configure environment:
   ```bash
   cp .env.example .env
   ```
   Fill in `TELEGRAM_BOT_TOKEN`.

   `OPENAI_API_KEY` is optional:
   - If it is set, the bot will generate daily summaries and answer `/ask` using OpenAI.
   - If it is missing, the bot still runs, but it will use a simplified daily summary and will tell you to add `OPENAI_API_KEY` before answering `/ask`.

3. Run the bot:
   ```bash
   python bot.py
   ```

4. Add the bot to your group chat(s) with colleagues.

## Important: Telegram "Privacy mode"

To analyze *all* incoming information, you typically must disable the bot's privacy mode:
- In BotFather, use `/setprivacy` and turn privacy mode off.

Commands like `/ask` will usually still work, but disabling privacy ensures the bot receives messages it needs for full daily summaries.

## Commands

- `/ask <question>`: Answer using chat context (latest daily summary + relevant recent messages).

If you send a plain message like “What can you do?” the bot will reply with instructions.

## Notes / Limitations

- Media processing is limited to metadata + captions (no OCR/transcription).
- Daily summaries are generated per chat where new messages were received during the last 24 hours.

## Deploy to server (universal script)

Added script: `scripts/deploy.sh`

What it does:
- Syncs current project to server via `rsync` (excluding `.git`, `.venv`, local `.env`).
- Creates/updates `.venv` on server.
- Installs `requirements.txt`.
- Restarts bot with your command.

One-time local setup:
```bash
chmod +x scripts/deploy.sh
```

Example deploy:
```bash
scripts/deploy.sh \
  --host ubuntu@203.0.113.10 \
  --path /opt/telegram-context-bot \
  --restart "systemctl restart telegram-context-bot"
```

If your service has another name, just change `--restart`.

### SSH-only update (no rsync)

If code is already on server and you want simple updates via SSH:

```bash
chmod +x scripts/update-via-ssh.sh
scripts/update-via-ssh.sh --host root@88.218.122.81
```

This command runs on server:
- `git fetch` + `git pull --ff-only`
- installs `requirements.txt` into `.venv`
- `systemctl restart telegram-context-bot`

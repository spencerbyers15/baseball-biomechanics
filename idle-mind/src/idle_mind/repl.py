"""Terminal REPL for Phase 1.

Commands:
    /stream        idle thoughts since your last message (raw)
    /digest        the last digest JSON
    /clock         recent clock marks
    /sleep N       simulate N seconds of idle time (fast-forward; SIMULATED)
    /budget        today's token usage vs. budget
    /watch         toggle live echo of idle thoughts (default on)
    /help          this list
    /quit          exit
Anything else is a message to the mind.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .app import Mind
from .clock import human_duration, local_stamp
from .config import Config

DIM = "\033[2m"
RESET = "\033[0m"


class StdinLines:
    """Async stdin lines. Reads through the event loop (ttys and pipes), so
    Ctrl-C cancels the pending read and nothing is left blocking interpreter
    exit — a thread blocked in input() would wedge shutdown. Regular-file
    stdin can't be epoll'd; it falls back to a thread read, which never
    blocks on a regular file anyway."""

    def __init__(self) -> None:
        self._reader: asyncio.StreamReader | None = None
        self._use_thread = False

    async def readline(self, prompt: str) -> str:
        print(prompt, end="", flush=True)
        if self._reader is None and not self._use_thread:
            try:
                loop = asyncio.get_running_loop()
                reader = asyncio.StreamReader()
                protocol = asyncio.StreamReaderProtocol(reader)
                await loop.connect_read_pipe(lambda: protocol, sys.stdin)
                self._reader = reader
            except (ValueError, OSError):
                self._use_thread = True
        if self._reader is not None:
            data = await self._reader.readline()
            if not data:
                raise EOFError
            return data.decode(errors="replace").rstrip("\n")
        line = await asyncio.to_thread(sys.stdin.readline)
        if not line:
            raise EOFError
        return line.rstrip("\n")


class Repl:
    def __init__(self, mind: Mind) -> None:
        self.mind = mind
        self.watch = True
        mind.idle.on_thought = self._on_thought

    def _on_thought(self, text: str, simulated: bool) -> None:
        if self.watch:
            tag = "idle·sim" if simulated else "idle"
            print(f"\n{DIM}({tag}) {text}{RESET}")

    async def run(self) -> None:
        print(__doc__)
        stdin = StdinLines()
        while True:
            try:
                line = (await stdin.readline("\nyou> ")).strip()
            except (EOFError, KeyboardInterrupt, asyncio.CancelledError):
                print()
                break
            if not line:
                continue
            if line.startswith("/"):
                if not await self.command(line):
                    break
            else:
                reply = await self.mind.handle_user_message(line)
                print(f"\nmind> {reply}")

    async def command(self, line: str) -> bool:
        parts = line.split(maxsplit=1)
        cmd, arg = parts[0].lower(), parts[1] if len(parts) > 1 else ""
        if cmd == "/quit":
            return False
        if cmd == "/help":
            print(__doc__)
        elif cmd == "/stream":
            thoughts = await self.mind.stream_since_last_turn()
            if not thoughts:
                print("(no idle thoughts since your last message)")
            for t in thoughts:
                tag = " [sim]" if t["simulated"] else ""
                print(f"[{local_stamp(t['ts'])}]{tag} #{t['tick_n']} {t['content']}\n")
        elif cmd == "/digest":
            digest = await self.mind.last_digest_dict()
            print(json.dumps(digest, indent=2) if digest else "(no digest yet)")
        elif cmd == "/clock":
            offset = self.mind.clock.offset
            suffix = f"(simulated offset: +{offset:.0f}s)" if offset else "(real)"
            print(f"mind time: {local_stamp(self.mind.clock.now())}  {suffix}")
            for m in await self.mind.clock_marks():
                tag = " [sim]" if m["simulated"] else ""
                print(f"[{local_stamp(m['ts'])}]{tag} {m['event']}")
        elif cmd == "/sleep":
            try:
                seconds = float(arg)
            except ValueError:
                print("usage: /sleep <seconds>")
                return True
            print(f"[sim] fast-forwarding {human_duration(seconds)}...")
            try:
                ticks = await self.mind.sleep_sim(seconds)
            except RuntimeError as e:
                print(f"[sim] cannot simulate: {e} (/budget for status)")
                return True
            print(f"[sim] advanced {seconds:.0f}s, ran {ticks} idle ticks "
                  "(marked simulated in db and logs)")
        elif cmd == "/budget":
            used, total = await self.mind.budget_status()
            paused = " — IDLE LOOP PAUSED" if self.mind.idle.budget_paused else ""
            print(f"tokens today: {used:,} / {total:,}{paused}")
        elif cmd == "/watch":
            self.watch = not self.watch
            print(f"watch {'on' if self.watch else 'off'}")
        else:
            print(f"unknown command {cmd}; /help for the list")
        return True


async def amain(args: argparse.Namespace) -> None:
    cfg = Config.load(args.config)
    mind = Mind.create(cfg, fake=args.fake, db_path=args.db)
    repl = Repl(mind)
    await mind.start()
    if args.fake:
        print("*** FAKE LLM MODE — canned outputs, no API calls ***")
    try:
        await repl.run()
    finally:
        await mind.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(prog="idle-mind")
    parser.add_argument("--config", default="config.toml", help="path to config.toml")
    parser.add_argument("--db", default=None, help="override db path")
    parser.add_argument(
        "--fake", action="store_true",
        help="use a deterministic fake LLM (no API key or tokens needed)",
    )
    args = parser.parse_args()
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()

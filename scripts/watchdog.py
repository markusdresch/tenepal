#!/usr/bin/env python3
"""
Tenepal Command Dispatcher + Agent Watchdog

Du schreibst Befehle -> gehen an CC -> CC delegiert an verfügbare Agents.
- CC startet automatisch mit --dangerously-skip-permissions
- Codex startet mit --full-auto
- Watchdog erkennt Yes/No Prompts und fragt CC was zu tun ist
- Cursor-Keys + Enter für Auswahl-Bestätigung
"""

import subprocess
import time
import threading
import logging
import re
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

STUCK_THRESHOLD    = 300    # sekunden bis "stuck"
CHECK_INTERVAL     = 10     # watchdog check interval (kurz wegen prompt-detection)
PROMPT_CHECK_LINES = 20     # wieviele letzte Zeilen auf Prompts prüfen
LOG_FILE = Path(__file__).parent.parent / "watchdog.log"

# Start-Commands pro Agent-Typ
AGENT_START_COMMANDS = {
    "cc-opus":   "claude --dangerously-skip-permissions",
    "cc-sonnet": "claude --dangerously-skip-permissions",
    "cc-haiku":  "claude --dangerously-skip-permissions",
    "codex":     "codex --full-auto",
    "mistral":   None,   # manuell starten
    "modal":     None,   # manuell starten
}

AGENT_PROFILES = {
    "cc-opus": {
        "patterns": ["claude", "cc", "opus"],
        "model": "claude-opus-4-5",
        "role": "planner",
        "cost": "high",
        "good_for": ["architecture", "planning", "review", "merge-decision"]
    },
    "cc-sonnet": {
        "patterns": ["sonnet", "claude-sonnet"],
        "model": "claude-sonnet-4-6",
        "role": "worker",
        "cost": "medium",
        "good_for": ["implementation", "analysis", "writing"]
    },
    "cc-haiku": {
        "patterns": ["haiku"],
        "model": "claude-haiku-4-5",
        "role": "quick",
        "cost": "low",
        "good_for": ["small tasks", "formatting", "quick checks"]
    },
    "codex": {
        "patterns": ["codex"],
        "model": "codex",
        "role": "coder",
        "cost": "medium",
        "good_for": ["coding", "second opinion", "refactoring"]
    },
    "mistral": {
        "patterns": ["mistral"],
        "model": "mistral-api",
        "role": "experimenter",
        "cost": "low",
        "good_for": ["experiments", "chaos", "prompt variants"]
    },
    "modal": {
        "patterns": ["modal"],
        "model": "modal-gpu",
        "role": "compute",
        "cost": "gpu",
        "good_for": ["audio processing", "model inference", "batch jobs"]
    },
}

FALLBACK_CHAIN = ["mistral", "codex", "cc-sonnet", "cc-haiku"]

# Prompt-Pattern die der Watchdog erkennen soll
# Format: (regex, beschreibung, default_action)
# default_action: "yes" | "yes_always" | "no" | "ask_cc"
PROMPT_PATTERNS = [
    # CC / Claude Code style
    (r"(yes|no|yes, don't ask again)", "CC Bestätigung",        "ask_cc"),
    (r"(Yes|No|Always)",               "CC Auswahl",            "ask_cc"),
    (r"\(y/n\)",                        "Ja/Nein Prompt",        "ask_cc"),
    (r"\(yes/no\)",                     "Yes/No Prompt",         "ask_cc"),
    (r"Allow|Deny|Skip",               "Permission Prompt",     "ask_cc"),
    (r"Press Enter to continue",        "Continue Prompt",       "yes"),
    (r"\[Y/n\]",                        "Default-Yes Prompt",    "yes"),
    (r"\[y/N\]",                        "Default-No Prompt",     "ask_cc"),
    # Codex / allgemein
    (r"Proceed\?",                      "Proceed Prompt",        "ask_cc"),
    (r"Are you sure",                   "Confirmation Prompt",   "ask_cc"),
    (r"Overwrite\?",                    "Overwrite Prompt",      "ask_cc"),
]

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("dispatcher")

# ── tmux helpers ──────────────────────────────────────────────────────────────

def list_tmux_sessions() -> list[dict]:
    try:
        result = subprocess.run(
            ["tmux", "list-panes", "-a",
             "-F", "#{session_name}:#{window_index}.#{pane_index} #{pane_current_command}"],
            capture_output=True, text=True
        )
        panes = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split(" ", 1)
            target = parts[0]
            cmd = parts[1] if len(parts) > 1 else ""
            session = target.split(":")[0]
            panes.append({"target": target, "session": session, "cmd": cmd})
        return panes
    except Exception as e:
        log.error(f"list_tmux_sessions: {e}")
        return []


def get_pane_content(target: str, lines: int = 0) -> str:
    """Liest Pane-Output. lines=0 = alles, sonst letzte N Zeilen."""
    try:
        cmd = ["tmux", "capture-pane", "-t", target, "-p"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        content = result.stdout
        if lines > 0:
            return "\n".join(content.strip().split("\n")[-lines:])
        return content
    except Exception:
        return ""


def send_to_pane(target: str, text: str, press_enter: bool = True):
    try:
        subprocess.run(
            ["tmux", "send-keys", "-t", target, text],
            check=True, capture_output=True
        )
        if press_enter:
            time.sleep(0.1)
            subprocess.run(
                ["tmux", "send-keys", "-t", target, "", "Enter"],
                capture_output=True
            )
        log.info(f"→ [{target}]: {text[:80]}")
    except subprocess.CalledProcessError:
        log.error(f"Pane '{target}' nicht erreichbar")
        print(f"\n[ERROR] Pane '{target}' nicht gefunden.")


def send_cursor_and_enter(target: str, direction: str = "up", steps: int = 0):
    """Sendet Cursor-Keys + Enter für Menü-Navigation."""
    key_map = {"up": "Up", "down": "Down", "left": "Left", "right": "Right"}
    key = key_map.get(direction, "Up")
    for _ in range(steps):
        subprocess.run(["tmux", "send-keys", "-t", target, f"[{key}]", ""], capture_output=True)
        time.sleep(0.05)
    subprocess.run(["tmux", "send-keys", "-t", target, "", "Enter"], capture_output=True)
    log.info(f"→ [{target}]: cursor {direction}x{steps} + Enter")


def select_option(target: str, option: str):
    """
    Wählt eine Option in einem interaktiven Prompt.
    option: "yes" | "yes_always" | "no" | "1" | "2" etc.
    """
    option = option.lower().strip()

    if option in ("yes", "y"):
        send_to_pane(target, "y")
    elif option in ("yes_always", "always", "a"):
        # "Yes, always" ist meist zweite Option -> Cursor Down + Enter
        # oder direkt tippen
        send_to_pane(target, "a", press_enter=False)
        time.sleep(0.1)
        send_to_pane(target, "", press_enter=True)
    elif option in ("no", "n"):
        send_to_pane(target, "n")
    elif option.isdigit():
        # Numerische Auswahl
        send_cursor_and_enter(target, "down", int(option) - 1)
    else:
        # Freitext
        send_to_pane(target, option)

# ── Prompt Detection ──────────────────────────────────────────────────────────

def detect_prompt(content: str) -> tuple[str, str] | None:
    """
    Prüft ob der Pane-Inhalt einen interaktiven Prompt enthält.
    Returns: (beschreibung, default_action) oder None
    """
    last_lines = "\n".join(content.strip().split("\n")[-PROMPT_CHECK_LINES:])
    for pattern, desc, action in PROMPT_PATTERNS:
        if re.search(pattern, last_lines, re.IGNORECASE):
            return desc, action
    return None


def ask_cc_about_prompt(cc_target: str, agent_name: str, prompt_desc: str, pane_content: str) -> str:
    """
    Fragt CC was bei einem Prompt zu tun ist.
    Gibt die gewählte Aktion zurück: "yes" | "yes_always" | "no"
    """
    context = "\n".join(pane_content.strip().split("\n")[-15:])
    question = (
        f"[DISPATCHER] Agent '{agent_name}' hat einen Prompt:\n"
        f"Typ: {prompt_desc}\n"
        f"Letzter Output:\n{context}\n\n"
        f"Antwort mit: YES / YES_ALWAYS / NO"
    )
    send_to_pane(cc_target, question)

    # Warten auf CC-Antwort (max 30s)
    time.sleep(5)
    cc_response = get_pane_content(cc_target, lines=5).upper()

    if "YES_ALWAYS" in cc_response or "ALWAYS" in cc_response:
        return "yes_always"
    elif "NO" in cc_response:
        return "no"
    else:
        return "yes"  # default

# ── Agent Discovery + Spawn ───────────────────────────────────────────────────

def discover_agents() -> dict[str, str]:
    panes = list_tmux_sessions()
    found = {}
    for agent_name, profile in AGENT_PROFILES.items():
        for pane in panes:
            session_lower = pane["session"].lower()
            if any(p in session_lower for p in profile["patterns"]):
                if agent_name not in found:
                    found[agent_name] = pane["target"]
                    break
    return found


def find_cc_primary(agents: dict[str, str]) -> str | None:
    for candidate in ["cc-opus", "cc-sonnet", "cc-haiku"]:
        if candidate in agents:
            return candidate
    return None


def find_fallback(agents: dict[str, str], unavailable: str) -> str | None:
    for agent in FALLBACK_CHAIN:
        if agent != unavailable and agent in agents:
            return agent
    return None


def spawn_agent(agent_name: str) -> str | None:
    """Startet einen Agent in einer neuen tmux-Session."""
    project_dir = Path(__file__).parent.parent
    start_cmd = AGENT_START_COMMANDS.get(agent_name)

    if not start_cmd:
        print(f"  [!] Kein Start-Command für '{agent_name}' definiert.")
        return None

    session_name = agent_name.replace("-", "")  # "cc-opus" -> "ccopus"

    try:
        existing = subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            capture_output=True
        )
        if existing.returncode == 0:
            subprocess.run([
                "tmux", "new-window", "-t", session_name,
                "-n", agent_name, "-c", str(project_dir)
            ], check=True, capture_output=True)
        else:
            subprocess.run([
                "tmux", "new-session", "-d",
                "-s", session_name,
                "-n", agent_name,
                "-c", str(project_dir)
            ], check=True, capture_output=True)

        target = f"{session_name}:{agent_name}.0"
        time.sleep(0.5)
        subprocess.run(["tmux", "send-keys", "-t", target, start_cmd, "Enter"])
        log.info(f"[{agent_name}] gestartet: {target} ({start_cmd})")
        print(f"  [+] {agent_name} gestartet: {target}")
        print(f"      Command: {start_cmd}")
        time.sleep(3)
        return target

    except subprocess.CalledProcessError as e:
        log.error(f"spawn_agent({agent_name}) failed: {e}")
        print(f"  [ERROR] {agent_name} konnte nicht gestartet werden")
        return None

# ── Watchdog ──────────────────────────────────────────────────────────────────

def watchdog_loop(agents_ref: list, cc_ref: list):
    """
    Überwacht alle Agent-Panes:
    - Stuck detection -> Enter senden
    - Prompt detection -> CC fragen, Auswahl bestätigen
    """
    state   = {}   # target -> (last_content, last_change_time)
    prompts = {}   # target -> last_prompt_time (cooldown)

    while True:
        time.sleep(CHECK_INTERVAL)

        current_agents = agents_ref[0] if agents_ref else {}
        cc_target      = cc_ref[0] if cc_ref else None

        for agent_name, target in current_agents.items():
            content = get_pane_content(target)
            now     = time.time()

            if target not in state:
                state[target] = (content, now)
                continue

            last_content, last_change = state[target]

            if content != last_content:
                state[target] = (content, now)

                # Prompt-Check auf neuem Content
                prompt_result = detect_prompt(content)
                if prompt_result:
                    prompt_desc, default_action = prompt_result
                    cooldown = prompts.get(target, 0)

                    if now - cooldown > 30:  # max alle 30s ein Prompt-Handle
                        prompts[target] = now
                        log.info(f"[{agent_name}] Prompt erkannt: {prompt_desc} (default: {default_action})")
                        print(f"\n  [PROMPT] {agent_name}: {prompt_desc}")

                        if default_action == "yes":
                            print(f"  [AUTO] Bestätige mit Yes")
                            select_option(target, "yes")
                        elif default_action == "ask_cc" and cc_target and cc_target != target:
                            action = ask_cc_about_prompt(cc_target, agent_name, prompt_desc, content)
                            print(f"  [CC→] Aktion: {action}")
                            select_option(target, action)
                        else:
                            print(f"  [MANUAL] Bitte manuell bestätigen: {prompt_desc}")
            else:
                # Stuck check
                stuck_for = now - last_change
                if stuck_for >= STUCK_THRESHOLD:
                    log.warning(f"[{agent_name}@{target}] stuck {int(stuck_for)}s")
                    print(f"\n  [STUCK] {agent_name} seit {int(stuck_for)}s - sende Enter")
                    send_to_pane(target, "", press_enter=True)
                    state[target] = (content, now)

# ── UI ────────────────────────────────────────────────────────────────────────

def print_help():
    print("""
╔══════════════════════════════════════════════════════════════╗
║  Tenepal Dispatcher                                      ║
╠══════════════════════════════════════════════════════════════╣
║  Befehle -> CC -> CC delegiert                               ║
║                                                              ║
║  :agents              - erkannte Agents                      ║
║  :rescan              - tmux neu scannen                     ║
║  :spawn AGENT         - Agent starten (cc-opus/codex/...)    ║
║  :send AGENT text     - direkt an Agent                      ║
║  :confirm AGENT opt   - Prompt manuell bestätigen            ║
║                         opt: yes/yes_always/no/1/2/...       ║
║  :status              - config + agents                      ║
║  :log                 - letzte 10 Log-Einträge               ║
║  :q / exit            - beenden                              ║
║                                                              ║
║  Enter (leer)  -> Enter an CC                                ║
║  alles andere  -> primärer CC                                ║
╚══════════════════════════════════════════════════════════════╝
""")


def print_agents(agents: dict[str, str]):
    if not agents:
        print("  [!] Keine Agents gefunden.")
        return
    print("  Erkannte Agents:")
    for name, target in agents.items():
        profile = AGENT_PROFILES[name]
        start   = AGENT_START_COMMANDS.get(name, "manuell")
        print(f"    {name:15} @ {target:25} [{profile['model']}]")
        print(f"    {'':15}   start: {start}")


def main():
    print_help()

    agents   = discover_agents()
    agents_ref = [agents]

    print_agents(agents)
    print()

    cc_name = find_cc_primary(agents)
    if cc_name:
        cc_target = agents[cc_name]
        print(f"  Primärer CC: {cc_name} @ {cc_target}\n")
    else:
        print("  [!] Kein CC gefunden - starte cc-opus automatisch...")
        cc_target = spawn_agent("cc-opus")
        if cc_target:
            agents = discover_agents()
            agents_ref[0] = agents
            cc_name = find_cc_primary(agents) or "cc-opus"
            agents[cc_name] = cc_target
        else:
            print("  [!] CC-Start fehlgeschlagen. :spawn cc-opus versuchen.\n")
        print()

    cc_ref = [cc_target]

    # Watchdog
    t = threading.Thread(target=watchdog_loop, args=(agents_ref, cc_ref), daemon=True)
    t.start()
    log.info("Dispatcher + Watchdog gestartet")

    # REPL
    while True:
        try:
            raw = input("→ ")
        except (EOFError, KeyboardInterrupt):
            print("\nDispatcher beendet.")
            break

        cmd = raw.strip()

        if cmd in (":q", "exit", "quit"):
            print("Dispatcher beendet.")
            break

        elif cmd == ":agents":
            print_agents(agents)

        elif cmd == ":rescan":
            agents = discover_agents()
            agents_ref[0] = agents
            cc_name   = find_cc_primary(agents)
            cc_target = agents.get(cc_name) if cc_name else None
            cc_ref[0] = cc_target
            print_agents(agents)

        elif cmd.startswith(":spawn "):
            agent = cmd.split(" ", 1)[1].strip()
            if agent not in AGENT_PROFILES:
                print(f"  Unbekannter Agent: {agent}. Verfügbar: {list(AGENT_PROFILES.keys())}")
            else:
                target = spawn_agent(agent)
                if target:
                    agents[agent] = target
                    agents_ref[0] = agents

        elif cmd.startswith(":confirm "):
            parts = cmd.split(" ", 2)
            if len(parts) < 3:
                print("  Usage: :confirm AGENT yes|yes_always|no|1|2")
            else:
                agent_name, option = parts[1], parts[2]
                if agent_name in agents:
                    select_option(agents[agent_name], option)
                    log.info(f"Manuell bestätigt: [{agent_name}] -> {option}")
                else:
                    print(f"  Agent '{agent_name}' nicht gefunden.")

        elif cmd.startswith(":send "):
            parts = cmd.split(" ", 2)
            if len(parts) < 3:
                print("  Usage: :send AGENT text")
            else:
                agent_name, text = parts[1], parts[2]
                if agent_name in agents:
                    send_to_pane(agents[agent_name], text)
                else:
                    print(f"  Agent '{agent_name}' nicht gefunden.")

        elif cmd == ":status":
            print(f"  CC: {cc_name} @ {cc_target}")
            print(f"  Stuck: {STUCK_THRESHOLD}s | Check: {CHECK_INTERVAL}s")
            print(f"  Agents: {list(agents.keys())}")
            print(f"  Log: {LOG_FILE}")

        elif cmd == ":log":
            if LOG_FILE.exists():
                lines = LOG_FILE.read_text().strip().split("\n")
                for line in lines[-10:]:
                    print(f"  {line}")

        elif cmd == "":
            if cc_target:
                send_to_pane(cc_target, "", press_enter=True)

        else:
            if cc_target:
                send_to_pane(cc_target, cmd)
            else:
                print("  [!] Kein CC. :spawn cc-opus versuchen.")


if __name__ == "__main__":
    main()

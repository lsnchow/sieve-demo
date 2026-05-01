"""Interactive prompt helpers with arrow-key support when available."""

from __future__ import annotations

from typing import Iterable

try:
    import questionary
except Exception:  # pragma: no cover - optional dependency
    questionary = None


class Prompter:
    """Unified prompting API with questionary first, stdin fallback."""

    def __init__(self):
        self._questionary = questionary

    @property
    def rich_mode(self) -> bool:
        return self._questionary is not None

    def select(self,
               message: str,
               choices: Iterable[tuple[str, str]],
               default: str | None = None) -> str:
        choice_list = list(choices)
        if not choice_list:
            raise ValueError("choices must not be empty")

        if self._questionary is not None:
            q_choices = [
                self._questionary.Choice(title=title, value=value)
                for title, value in choice_list
            ]
            result = self._questionary.select(
                message=message,
                choices=q_choices,
                default=default,
                qmark=">",
            ).ask()
            if result is None:
                # e.g. Ctrl+C or cancelled prompt
                return default or choice_list[0][1]
            return str(result)

        print(f"\n{message}")
        for i, (title, _value) in enumerate(choice_list, 1):
            print(f"  {i}. {title}")
        while True:
            raw = input("Select option: ").strip()
            if not raw and default is not None:
                return default
            if raw.isdigit():
                idx = int(raw) - 1
                if 0 <= idx < len(choice_list):
                    return choice_list[idx][1]
            lowered = raw.lower()
            for title, value in choice_list:
                if lowered in (title.lower(), value.lower()):
                    return value
            print("Invalid choice. Try again.")

    def text(self,
             message: str,
             default: str = "",
             allow_empty: bool = False) -> str:
        if self._questionary is not None:
            result = self._questionary.text(
                message=message,
                default=default,
                qmark=">",
            ).ask()
            result = default if result is None else str(result)
        else:
            suffix = f" [{default}]" if default else ""
            result = input(f"{message}{suffix}: ").strip()
            if not result:
                result = default

        if result or allow_empty:
            return result
        return self.text(message, default=default, allow_empty=allow_empty)

    def confirm(self, message: str, default: bool = True) -> bool:
        if self._questionary is not None:
            result = self._questionary.confirm(
                message=message,
                default=default,
                qmark=">",
            ).ask()
            return bool(default if result is None else result)

        yn = "Y/n" if default else "y/N"
        raw = input(f"{message} [{yn}]: ").strip().lower()
        if not raw:
            return default
        return raw in ("y", "yes")


def get_prompter() -> Prompter:
    return Prompter()


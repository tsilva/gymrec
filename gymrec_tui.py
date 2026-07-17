"""Textual presentation layer for Gymrec's interactive selectors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.content import Content
from textual.fuzzy import FuzzySearch
from textual.widgets import Footer, Input, Label, OptionList, Static
from textual.widgets.option_list import Option


class SelectorItem(Protocol):
    """Minimal display contract supplied by Gymrec's CLI layer."""

    key: str
    label: str
    category: str
    search_text: str


class GymrecSelectorApp(App[str]):
    """Search and select one opaque key without owning domain semantics."""

    ENABLE_COMMAND_PALETTE = False

    CSS = """
    Screen {
        align: center middle;
        background: $background;
    }

    #selector {
        width: 92%;
        max-width: 112;
        height: 88%;
        min-height: 16;
        padding: 1 2;
        border: round $primary;
        background: $surface;
    }

    #title {
        height: 2;
        content-align: left middle;
        text-style: bold;
        color: $text-accent;
    }

    #search {
        margin-bottom: 1;
    }

    #results {
        height: 1fr;
        border: round $panel;
        padding: 0 1;
    }

    #status {
        height: 2;
        padding: 0 1;
        content-align: left middle;
        color: $text-muted;
    }

    OptionList > .option-list--option-highlighted {
        text-style: bold;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("slash", "focus_search", "Search", key_display="/", priority=True),
    ]

    def __init__(
        self,
        items: Sequence[SelectorItem],
        *,
        title: str,
        placeholder: str,
    ) -> None:
        super().__init__()
        self._items = tuple(items)
        self._title = title
        self._placeholder = placeholder
        self._visible_items = self._items
        self._fuzzy = FuzzySearch(case_sensitive=False)

    def compose(self) -> ComposeResult:
        with Vertical(id="selector"):
            yield Label(self._title, id="title")
            yield Input(placeholder=self._placeholder, id="search")
            yield OptionList(id="results", markup=False)
            yield Static(id="status")
            yield Footer()

    def on_mount(self) -> None:
        self._apply_filter("")
        self.query_one("#search", Input).focus()

    def _prompt(self, item: SelectorItem, positions: Sequence[int]) -> Content:
        display = f"{item.category}  {item.label}"
        prompt = Content(display)
        prompt = prompt.stylize("bold #5fd7ff", 0, len(item.category))
        for position in positions:
            prompt = prompt.stylize("reverse bold", position, position + 1)
        return prompt

    def _apply_filter(self, raw_query: str) -> None:
        query = raw_query.strip()
        results = self.query_one("#results", OptionList)
        highlighted_key = None
        if results.highlighted_option is not None:
            highlighted_key = results.highlighted_option.id

        if not query:
            ranked = [(item, ()) for item in self._items]
        else:
            scored = []
            for order, item in enumerate(self._items):
                display = f"{item.category}  {item.label}"
                score, _search_positions = self._fuzzy.match(query, item.search_text)
                if score > 0:
                    # Match positions are relative to search_text. Re-run against the
                    # literal display string so highlight offsets remain safe and exact.
                    _display_score, display_positions = self._fuzzy.match(query, display)
                    scored.append((score, order, item, display_positions))
            scored.sort(key=lambda value: (-value[0], value[1]))
            ranked = [(item, positions) for _score, _order, item, positions in scored]

        self._visible_items = tuple(item for item, _positions in ranked)
        results.set_options(
            Option(self._prompt(item, positions), id=item.key)
            for item, positions in ranked
        )

        visible_keys = [item.key for item in self._visible_items]
        if highlighted_key in visible_keys:
            results.highlighted = visible_keys.index(highlighted_key)
        elif visible_keys:
            results.highlighted = 0
        else:
            results.highlighted = None

        status = self.query_one("#status", Static)
        if not visible_keys:
            status.update("No matches — refine your search or press Esc to cancel")
        elif query:
            status.update(f"{len(visible_keys)} of {len(self._items)} matches")
        else:
            status.update(f"{len(visible_keys)} available")

    @on(Input.Changed, "#search")
    def _search_changed(self, event: Input.Changed) -> None:
        self._apply_filter(event.value)

    @on(Input.Submitted, "#search")
    def _search_submitted(self) -> None:
        results = self.query_one("#results", OptionList)
        if results.highlighted_option is not None:
            self.exit(results.highlighted_option.id)

    @on(OptionList.OptionSelected, "#results")
    def _option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id is not None:
            self.exit(event.option.id)

    def on_key(self, event: events.Key) -> None:
        if event.key == "down" and isinstance(self.focused, Input):
            results = self.query_one("#results", OptionList)
            if results.option_count:
                results.focus()
                event.prevent_default()
                event.stop()

    def action_focus_search(self) -> None:
        self.query_one("#search", Input).focus()

    def action_cancel(self) -> None:
        self.exit(None)


async def select_item(
    items: Sequence[SelectorItem],
    *,
    title: str,
    placeholder: str,
) -> str | None:
    """Run the full-screen selector in the caller's existing asyncio loop."""
    return await GymrecSelectorApp(
        items,
        title=title,
        placeholder=placeholder,
    ).run_async()

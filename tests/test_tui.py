import asyncio
from dataclasses import dataclass

from textual.widgets import Input, OptionList, Static

import gymrec_tui


@dataclass(frozen=True)
class Item:
    key: str
    category: str
    label: str
    search_text: str


def make_app(items=None):
    return gymrec_tui.GymrecSelectorApp(
        items
        or [
            Item("sonic", "Stable-Retro", "SonicTheHedgehog-Genesis-v0", "Stable-Retro SonicTheHedgehog-Genesis-v0"),
            Item("mario", "Mario Turbo", "SuperMarioBros-Nes-v0", "Mario Turbo SuperMarioBros-Nes-v0"),
        ],
        title="Select an environment",
        placeholder="Search environments",
    )


def test_selector_fuzzy_search_highlights_and_selects():
    async def run():
        app = make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            assert isinstance(app.focused, Input)
            assert app.query_one("#results", OptionList).option_count == 2

            await pilot.press("m", "a", "r", "i", "o")
            await pilot.pause()
            results = app.query_one("#results", OptionList)
            assert results.option_count == 1
            assert results.highlighted_option.id == "mario"
            assert "reverse" in repr(results.highlighted_option.prompt)
            assert str(app.query_one("#status", Static).render()) == "1 of 2 matches"

            await pilot.press("enter")
        assert app.return_value == "mario"

    asyncio.run(run())


def test_selector_preserves_highlight_across_refilter_and_supports_down():
    async def run():
        items = [
            Item("alpha", "Provider", "Alpha", "Provider Alpha"),
            Item("alpine", "Provider", "Alpine", "Provider Alpine"),
            Item("beta", "Provider", "Beta", "Provider Beta"),
        ]
        app = make_app(items)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            results = app.query_one("#results", OptionList)
            results.highlighted = 1
            app.query_one("#search", Input).value = "alp"
            await pilot.pause()
            assert results.highlighted_option.id == "alpine"

            app.query_one("#search", Input).focus()
            await pilot.press("down")
            assert app.focused is results
            await pilot.press("/")
            assert isinstance(app.focused, Input)
            assert app.query_one("#search", Input).value == "alp"

    asyncio.run(run())


def test_selector_keeps_duplicate_labels_distinct_and_mouse_selects():
    async def run():
        items = [
            Item("stable", "Stable-Retro", "SuperMarioBros-Nes-v0", "Stable-Retro SuperMarioBros-Nes-v0"),
            Item("turbo", "Mario Turbo", "SuperMarioBros-Nes-v0", "Mario Turbo SuperMarioBros-Nes-v0"),
        ]
        app = make_app(items)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            results = app.query_one("#results", OptionList)
            assert [option.id for option in results.options] == ["stable", "turbo"]
            assert await pilot.click("#results", offset=(5, 2))
            await pilot.pause()
        assert app.return_value == "turbo"

    asyncio.run(run())


def test_selector_no_matches_cannot_select_and_escape_cancels():
    async def run():
        app = make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press("z", "z", "z", "z")
            await pilot.pause()
            results = app.query_one("#results", OptionList)
            assert results.option_count == 0
            assert "No matches" in str(app.query_one("#status", Static).render())
            await pilot.press("enter")
            assert app.is_running
            await pilot.press("escape")
        assert app.return_value is None

    asyncio.run(run())


def test_selector_handles_thousand_item_catalog():
    async def run():
        items = [
            Item(
                f"item-{index}",
                "Stable-Retro",
                f"SyntheticGame{index:04d}-Nes-v0",
                f"Stable-Retro SyntheticGame{index:04d}-Nes-v0",
            )
            for index in range(1000)
        ]
        app = make_app(items)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            assert app.query_one("#results", OptionList).option_count == 1000
            app.query_one("#search", Input).value = "0999"
            await pilot.pause()
            results = app.query_one("#results", OptionList)
            assert results.option_count == 1
            assert results.highlighted_option.id == "item-999"

    asyncio.run(run())

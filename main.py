def make_label(text, size=20, height=45):
    return Label(
        text=text, font_size=sp(size * 1.3),
        size_hint_y=None, height=dp(height * 1.3),
    )


def make_button(text, callback, size=20, height=60, **kw):
    b = Button(
        text=text, font_size=sp(size * 1.3),
        size_hint_y=None, height=dp(height * 1.3), **kw
    )
    b.bind(on_press=callback)
    return b


def make_scroll_grid():
    sv = ScrollView()
    grid = GridLayout(
        cols=3, spacing=dp(8), padding=dp(8), size_hint_y=None
    )
    grid.bind(minimum_height=grid.setter("height"))
    sv.add_widget(grid)
    return sv, grid


def add_thumb(grid, path, on_delete=None):
    box = BoxLayout(
        orientation="vertical", size_hint=(None, None),
        size=(dp(105), dp(145)), spacing=dp(4),
    )
    box.add_widget(Image(
        source=path, size_hint=(None, None), size=(dp(105), dp(105))
    ))
    if on_delete:
        box.add_widget(
            make_button("Видалити", lambda i: on_delete(), 12, 26)
        )
    grid.add_widget(box)

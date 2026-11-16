"""Render a synthetic browser page from a declarative state.

The evaluation corpus needs recordings whose ground truth is known exactly, which means
generating them rather than annotating them. What is generated has to stay honest,
though: a corpus of large flat colour blocks would be trivially easy and would prove
nothing about a real capture.

So this renders at screen-capture resolution with real text at real sizes, which keeps
the property that actually stresses detection — a changed field occupies a fraction of
a percent of the frame, not a quarter of it.
"""

from PIL import Image, ImageDraw, ImageFont

WIDTH = 1920
HEIGHT = 1080

_FONT_DIR = "/System/Library/Fonts/Supplemental"
_BODY = ImageFont.truetype(f"{_FONT_DIR}/Arial.ttf", 24)
_BOLD = ImageFont.truetype(f"{_FONT_DIR}/Arial Bold.ttf", 24)
_HEADING = ImageFont.truetype(f"{_FONT_DIR}/Arial Bold.ttf", 46)
_SMALL = ImageFont.truetype(f"{_FONT_DIR}/Arial.ttf", 20)

_PAGE_BG = (255, 255, 255)
_CHROME_BG = (237, 239, 241)
_NAV_BG = (250, 251, 252)
_TEXT = (32, 33, 36)
_MUTED = (95, 99, 104)
_FIELD_BORDER = (188, 192, 196)
_ACCENT = (26, 115, 232)
_DANGER = (198, 40, 40)
_SUCCESS = (30, 120, 60)

_CONTENT_X = 560
_CONTENT_W = 800
_CHROME_H = 96
_NAV_H = 62


class Field:
    """One labelled input on the page."""

    def __init__(
        self,
        label: str,
        value: str = "",
        *,
        masked: bool = False,
        focused: bool = False,
    ) -> None:
        self.label = label
        self.value = value
        self.masked = masked
        self.focused = focused

    @property
    def display(self) -> str:
        return "•" * len(self.value) if self.masked else self.value


class PageState:
    """One rendered moment of the fake application.

    Every field is a plain value so a case file can describe a timeline as a list of
    states without needing to know how any of it is drawn.
    """

    def __init__(
        self,
        *,
        url: str = "app.example/login",
        heading: str = "Login Page",
        nav: list[str] | None = None,
        active_nav: str | None = None,
        banner: str | None = None,
        banner_kind: str = "error",
        fields: list[Field] | None = None,
        button: str | None = None,
        checkbox: bool | None = None,
        checkbox_label: str = "Remember me",
        modal: str | None = None,
        modal_body: str = "This action cannot be undone.",
        dropdown: list[str] | None = None,
        toast: str | None = None,
        rows: int = 0,
        row_prefix: str = "Order",
        scroll_y: int = 0,
        spinner_phase: int | None = None,
    ) -> None:
        self.url = url
        self.heading = heading
        self.nav = nav
        self.active_nav = active_nav
        self.banner = banner
        self.banner_kind = banner_kind
        self.fields = fields or []
        self.button = button
        self.checkbox = checkbox
        self.checkbox_label = checkbox_label
        self.modal = modal
        self.modal_body = modal_body
        self.dropdown = dropdown
        self.toast = toast
        self.rows = rows
        self.row_prefix = row_prefix
        self.scroll_y = scroll_y
        self.spinner_phase = spinner_phase


def _draw_chrome(draw: ImageDraw.ImageDraw, state: PageState) -> int:
    draw.rectangle([0, 0, WIDTH, _CHROME_H], fill=_CHROME_BG)
    draw.rounded_rectangle([180, 26, WIDTH - 180, 70], radius=22, fill=(255, 255, 255))
    draw.text((210, 36), state.url, font=_BODY, fill=_TEXT)
    if not state.nav:
        return _CHROME_H

    draw.rectangle([0, _CHROME_H, WIDTH, _CHROME_H + _NAV_H], fill=_NAV_BG)
    x = _CONTENT_X
    for item in state.nav:
        width = int(draw.textlength(item, font=_BOLD)) + 48
        active = item == state.active_nav
        draw.text((x + 24, _CHROME_H + 18), item, font=_BOLD if active else _BODY,
                  fill=_ACCENT if active else _MUTED)
        if active:
            draw.line([x + 20, _CHROME_H + _NAV_H - 4, x + width - 20, _CHROME_H + _NAV_H - 4],
                      fill=_ACCENT, width=4)
        x += width
    return _CHROME_H + _NAV_H


def _draw_field(draw: ImageDraw.ImageDraw, field: Field, y: int) -> int:
    draw.text((_CONTENT_X, y), field.label, font=_BOLD, fill=_TEXT)
    box = [_CONTENT_X, y + 34, _CONTENT_X + _CONTENT_W, y + 84]
    draw.rectangle(box, fill=(255, 255, 255),
                   outline=_ACCENT if field.focused else _FIELD_BORDER,
                   width=2 if field.focused else 1)
    text = field.display
    if text:
        draw.text((_CONTENT_X + 14, y + 46), text, font=_BODY, fill=_TEXT)
    if field.focused:
        caret_x = _CONTENT_X + 16 + int(draw.textlength(text, font=_BODY))
        draw.line([caret_x, y + 44, caret_x, y + 74], fill=_TEXT, width=2)
    return y + 108


def _draw_banner(draw: ImageDraw.ImageDraw, state: PageState, y: int) -> int:
    colour = _DANGER if state.banner_kind == "error" else _SUCCESS
    draw.rectangle([_CONTENT_X, y, _CONTENT_X + _CONTENT_W, y + 56], fill=colour)
    draw.text((_CONTENT_X + 20, y + 14), state.banner, font=_BOLD, fill=(255, 255, 255))
    return y + 84


def _draw_rows(draw: ImageDraw.ImageDraw, state: PageState, y: int) -> None:
    """Draw a scrollable result table; scroll_y shifts which rows are visible."""
    top = y - state.scroll_y
    for index in range(state.rows):
        row_y = top + index * 54
        # Rows are clipped to the viewport below the heading, as a real scroll region is.
        if row_y < y or row_y > HEIGHT - 54:
            continue
        if index % 2 == 0:
            draw.rectangle([_CONTENT_X, row_y, _CONTENT_X + _CONTENT_W, row_y + 52],
                           fill=(246, 248, 250))
        draw.text((_CONTENT_X + 16, row_y + 14), f"{state.row_prefix} #{4200 + index}",
                  font=_BODY, fill=_TEXT)
        draw.text((_CONTENT_X + 420, row_y + 14), f"£{12 + index * 7}.00", font=_BODY, fill=_MUTED)


def _draw_modal(image: Image.Image, state: PageState) -> None:
    # The scrim must be composited, not filled: an RGBA fill on an RGB canvas silently
    # drops the alpha and paints the page solid black.
    scrim = Image.new("RGBA", (WIDTH, HEIGHT - _CHROME_H), (20, 20, 20, 90))
    image.paste(Image.alpha_composite(
        image.crop((0, _CHROME_H, WIDTH, HEIGHT)).convert("RGBA"), scrim
    ).convert("RGB"), (0, _CHROME_H))

    draw = ImageDraw.Draw(image)
    draw.rectangle([660, 380, 1260, 700], fill=(255, 255, 255), outline=_FIELD_BORDER, width=2)
    draw.text((690, 420), state.modal, font=_BOLD, fill=_TEXT)
    draw.text((690, 480), state.modal_body, font=_BODY, fill=_MUTED)
    draw.rounded_rectangle([690, 590, 850, 650], radius=6, fill=_DANGER)
    draw.text((722, 608), "Confirm", font=_BOLD, fill=(255, 255, 255))
    draw.rounded_rectangle([880, 590, 1010, 650], radius=6, outline=_FIELD_BORDER, width=2)
    draw.text((908, 608), "Cancel", font=_BODY, fill=_TEXT)


def _draw_dropdown(draw: ImageDraw.ImageDraw, state: PageState, y: int) -> None:
    for index, option in enumerate(state.dropdown):
        option_y = y + index * 46
        draw.rectangle([_CONTENT_X, option_y, _CONTENT_X + _CONTENT_W, option_y + 46],
                       fill=(255, 255, 255), outline=_FIELD_BORDER)
        draw.text((_CONTENT_X + 14, option_y + 10), option, font=_BODY, fill=_TEXT)


def _draw_spinner(draw: ImageDraw.ImageDraw, phase: int) -> None:
    """Draw a rotating loading indicator — deliberate visual noise with no semantics."""
    cx, cy, r = _CONTENT_X + _CONTENT_W // 2, 760, 26
    start = (phase * 30) % 360
    draw.arc([cx - r, cy - r, cx + r, cy + r], start=start, end=start + 270, fill=_ACCENT, width=6)


def render(state: PageState) -> Image.Image:
    """Render one page state to a full-resolution frame."""
    image = Image.new("RGB", (WIDTH, HEIGHT), _PAGE_BG)
    draw = ImageDraw.Draw(image)
    y = _draw_chrome(draw, state) + 74

    if state.banner:
        y = _draw_banner(draw, state, y)

    draw.text((_CONTENT_X, y), state.heading, font=_HEADING, fill=_TEXT)
    y += 90

    if state.rows:
        _draw_rows(draw, state, y)
        if state.modal:
            _draw_modal(image, state)
        if state.toast:
            _draw_toast(draw, state)
        return image

    for field in state.fields:
        y = _draw_field(draw, field, y)

    if state.dropdown:
        _draw_dropdown(draw, state, y)
    else:
        if state.checkbox is not None:
            draw.rectangle([_CONTENT_X, y + 6, _CONTENT_X + 26, y + 32],
                           fill=_ACCENT if state.checkbox else (255, 255, 255),
                           outline=_FIELD_BORDER, width=2)
            if state.checkbox:
                draw.line([_CONTENT_X + 5, y + 19, _CONTENT_X + 11, y + 26],
                          fill=(255, 255, 255), width=3)
                draw.line([_CONTENT_X + 11, y + 26, _CONTENT_X + 21, y + 11],
                          fill=(255, 255, 255), width=3)
            draw.text((_CONTENT_X + 40, y + 6), state.checkbox_label, font=_SMALL, fill=_MUTED)
            y += 54
        if state.button:
            width = int(draw.textlength(state.button, font=_BOLD)) + 92
            draw.rounded_rectangle([_CONTENT_X, y + 6, _CONTENT_X + width, y + 62],
                                   radius=6, fill=_ACCENT)
            draw.text((_CONTENT_X + 46, y + 22), state.button, font=_BOLD, fill=(255, 255, 255))

    if state.spinner_phase is not None:
        _draw_spinner(draw, state.spinner_phase)
    if state.modal:
        _draw_modal(image, state)
    if state.toast:
        _draw_toast(ImageDraw.Draw(image), state)

    return image


def _draw_toast(draw: ImageDraw.ImageDraw, state: PageState) -> None:
    draw.rounded_rectangle([1420, 940, 1880, 1020], radius=8, fill=(38, 42, 46))
    draw.text((1446, 966), state.toast, font=_BODY, fill=(255, 255, 255))


def draw_cursor(frame: Image.Image, x: int, y: int) -> None:
    """Stamp a mouse pointer onto a frame.

    The cursor moves every frame, which is the point: it reproduces the low-level
    churn a real capture always carries, so the corpus tests that detection
    discriminates against a real noise floor rather than against a perfectly still image.
    """
    draw = ImageDraw.Draw(frame)
    draw.polygon(
        [(x, y), (x, y + 26), (x + 7, y + 19), (x + 12, y + 30),
         (x + 17, y + 27), (x + 12, y + 17), (x + 20, y + 17)],
        fill=(20, 20, 20),
        outline=(255, 255, 255),
    )

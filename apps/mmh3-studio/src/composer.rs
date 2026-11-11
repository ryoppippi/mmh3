use mmh3_studio::sizes::{PRESETS, megapixels};
use rune::{El, Key, KeyEvent, Tx, col, row, text};
use rune_kit::{Button, ButtonVariant, ControlSize, IconButton, IconButtonVariant, Kit, TextArea, TextInput};

use crate::icons;
use crate::state::{SizeChoice, State};

pub const GENERATE_LABEL: &str = "⌘↩";

pub fn is_generate(ev: &KeyEvent) -> bool {
    ev.key == Key::Enter && ev.mods.cmd
}

fn section(title: &str, k: &Kit) -> El {
    text(title.to_owned()).font(k.ty.small_medium).color(k.c.text_faint)
}

fn tile(s: &State, i: Option<usize>, k: &Kit) -> El {
    let c = &k.c;
    let (choice, title, detail) = match i {
        Some(i) => {
            let p = PRESETS[i];
            (SizeChoice::Preset(i), format!("{} × {}", p.width, p.height), format!("{} · {} · {:.1} MP", p.tier, p.aspect, megapixels(p.width, p.height)))
        }
        None => (SizeChoice::Custom, "Custom".to_owned(), "Multiples of 32".to_owned()),
    };
    let on = s.size == choice;
    let key = match i {
        Some(i) => format!("size-{i}"),
        None => "size-custom".to_owned(),
    };
    let t = col()
        .key(key)
        .role("radio")
        .label(format!("{title}, {detail}"))
        .focusable(true)
        .focus_selected(on)
        .w_full()
        .px(10.0)
        .py(8.0)
        .gap(2.0)
        .radius(8.0)
        .border(if on { 1.5 } else { 1.0 }, if on { c.border_focus } else { c.border_base })
        .fill(if on { c.layer_02 } else { c.bg_base })
        .on_press(move |s: &mut State| s.size = choice)
        .child(text(title).font(k.ty.ui_medium).color(c.text_base).a11y_hidden())
        .child(text(detail).font(k.ty.small).color(c.text_muted).truncate().a11y_hidden());
    k.interactive(t)
}

fn sizes(s: &State, k: &Kit) -> El {
    let mut grid = row().key("sizes").role("radiogroup").focus_group().label("Size").w_full().equal_columns(2).gaps(8.0, 8.0);
    let n = PRESETS.len() + 1;
    for i in 0..PRESETS.len() {
        grid = grid.child(tile(s, Some(i), k).position_in_set(i + 1, n));
    }
    grid = grid.child(tile(s, None, k).position_in_set(n, n));
    let mut out = col().w_full().gap(8.0).child(section("Size", k)).child(grid);
    if s.size == SizeChoice::Custom {
        let error = s.chosen_size().err();
        let field = |edit: &rune::TextEdit, label: &str, key: &str, lens: fn(&mut State) -> &mut rune::TextEdit| {
            TextInput::new(edit)
                .label(label)
                .placeholder("1024")
                .invalid(error.is_some())
                .on_edit(lens)
                .build(k)
                .key(key.to_owned())
                .grow(1.0)
                .shrink()
                .min_w(0.0)
        };
        out = out.child(
            row()
                .key("custom-size")
                .w_full()
                .gap(8.0)
                .center_y()
                .child(field(&s.custom_w, "Width", "custom-width", |s| &mut s.custom_w))
                .child(text("×").font(k.ty.ui).color(k.c.text_faint))
                .child(field(&s.custom_h, "Height", "custom-height", |s| &mut s.custom_h)),
        );
        let note = match &error {
            Some(e) => text(e.to_string()).key("size-error").role("alert").font(k.ty.small).color(k.c.danger.fg).wrap(),
            None => {
                let (w, h) = s.chosen_size().unwrap_or_default();
                text(format!("{:.1} MP", megapixels(w, h))).key("size-note").font(k.ty.small).color(k.c.text_faint)
            }
        };
        out = out.child(note);
    }
    out
}

fn seed(s: &State, k: &Kit) -> El {
    let invalid = s.seed_value().is_none();
    let field = TextInput::new(&s.seed)
        .label("Seed")
        .placeholder("Random")
        .invalid(invalid)
        .on_edit(|s: &mut State| &mut s.seed)
        .build(k)
        .key("seed")
        .grow(1.0)
        .shrink()
        .min_w(0.0);
    let dice = IconButton::new(&icons::DICE)
        .variant(IconButtonVariant::Ghost)
        .control_size(ControlSize::Large)
        .label("Randomise seed")
        .help("Randomise seed")
        .on_press(|s: &mut State| s.randomise_seed())
        .build(k)
        .key("randomise-seed");
    let mut out = col().w_full().gap(8.0).child(section("Seed", k)).child(row().w_full().gap(6.0).center_y().child(field).child(dice));
    if invalid {
        out = out.child(text("The seed must be a whole number.").key("seed-error").font(k.ty.small).color(k.c.danger.fg));
    }
    out
}

pub fn composer(s: &State, k: &Kit) -> El {
    let prompt = TextArea::new(&s.prompt)
        .label("Prompt")
        .placeholder("Describe the image…")
        .width(None)
        .auto_grow(5, 14)
        .on_edit(|s: &mut State| &mut s.prompt)
        .on_key(|s: &mut State, ev: &KeyEvent, _tx: &mut Tx| {
            if is_generate(ev) {
                s.generate();
                return true;
            }
            if ev.key == Key::Enter && !ev.mods.cmd && !ev.mods.ctrl && !ev.mods.alt && !ev.mods.shift {
                s.prompt.insert("\n");
                return true;
            }
            false
        })
        .build(k)
        .key("prompt");
    let ready = s.request();
    let help = match &ready {
        Ok(_) => "Generate an image".to_owned(),
        Err(why) => why.clone(),
    };
    let generate = Button::new("Generate")
        .variant(ButtonVariant::Submit)
        .control_size(ControlSize::Large)
        .icon(&icons::SPARKLE)
        .shortcut(GENERATE_LABEL)
        .help(help)
        .disabled(ready.is_err())
        .on_press(|s: &mut State| s.generate())
        .build(k)
        .key("generate")
        .w_full();
    let active = s.active_count();
    let queue = match active {
        0 => "Nothing queued. Generate as often as you like; requests queue on the server.".to_owned(),
        1 => "1 generation in progress.".to_owned(),
        n => format!("{n} generations queued or running."),
    };
    col()
        .key("composer")
        .w_full()
        .h_full()
        .min_h(0.0)
        .padding(16.0)
        .gap(18.0)
        .scroll(rune::ScrollSpec { stick_bottom: false })
        .child(col().w_full().gap(8.0).child(section("Prompt", k)).child(prompt))
        .child(sizes(s, k))
        .child(seed(s, k))
        .child(col().w_full().gap(8.0).child(generate).child(text(queue).key("queue-note").font(k.ty.small).color(k.c.text_faint).wrap()))
}

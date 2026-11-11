use std::sync::OnceLock;

use rune::{Color, Cx, El, Spring, col, el, row, text};
use rune_kit::shell::{self, PaneMotion, SidebarToggle, SplitView, Toolbar};
use rune_kit::{Button, ButtonVariant, ControlSize, Dialog, IconButton, IconButtonVariant, Kit, Toast, ToastRegion, context_menu};

use crate::state::{Conn, Phase, State};
use crate::{composer, gallery, icons, preview};

pub fn kit(dark: bool) -> &'static Kit {
    static LIGHT: OnceLock<Kit> = OnceLock::new();
    static DARK: OnceLock<Kit> = OnceLock::new();
    if dark { DARK.get_or_init(|| Kit::neutral(true)) } else { LIGHT.get_or_init(|| Kit::neutral(false)) }
}

pub fn view(s: &State, cx: &mut Cx<State>) -> El {
    let dark = cx.system_dark();
    let k = kit(dark);
    let mut layer = Vec::new();
    let main = preview::main(s, k, cx);
    let split = SplitView::new(main)
        .sidebar(&s.sidebar, |s: &mut State| &mut s.sidebar, gallery::sidebar(s, k, dark))
        .inspector(&s.inspector, |s: &mut State| &mut s.inspector, composer::composer(s, k))
        .toolbar(toolbar(s, k))
        .titlebar_inset(cx.titlebar_inset())
        .frame_inset(8.0)
        .toolbar_in_panel()
        .sidebar_padding(18.0, 10.0)
        .sidebar_toggle(SidebarToggle::new().size(ControlSize::Large).past_lights(16.0).edge(18.0).gap(4.0).in_header())
        .pane_motion(PaneMotion::new().spring(Spring::new(0.3, 0.0)).scale_with_width(480.0, 0.15, 0.3).fade_tail(0.15))
        .reduced_motion(cx.is_reduced_motion())
        .build(k);
    let split = menu(s, k, split, &mut layer);
    confirm_delete(s, k, &mut layer);
    if !s.notes.is_empty() {
        layer.push(notes(s, k));
    }
    let body =
        col().key("window").w_full().h_full().fill(k.c.page).child(col().key("split").w_full().grow(1.0).min_h(0.0).child(split)).child(status_bar(s, k));
    rune_kit::root(body, layer)
}

fn toolbar(s: &State, k: &Kit) -> Toolbar {
    let title = s.selected_gen().map_or_else(|| "mmh3 studio".to_owned(), |g| g.prompt.lines().next().unwrap_or_default().to_owned());
    let mut actions = row().key("toolbar-actions").gap(2.0).center_y();
    if let Some(g) = s.selected_gen() {
        let done = g.file().is_some();
        let running = matches!(g.phase, Phase::Running { .. } | Phase::Sending);
        let id = g.id.clone();
        type Action = Box<dyn Fn(&mut State, &mut rune::Tx)>;
        let act = |icon: &'static rune::Icon, key: &str, label: &str, disabled: bool, f: Action| {
            IconButton::new(icon)
                .variant(IconButtonVariant::GhostMuted)
                .control_size(ControlSize::Large)
                .label(label)
                .help(label)
                .disabled(disabled)
                .on_press_tx(move |s: &mut State, tx: &mut rune::Tx| f(s, tx))
                .build(k)
                .key(key.to_owned())
        };
        let i = id.clone();
        actions = actions.child(act(&icons::SAVE, "save", "Save PNG…", !done, Box::new(move |s, tx| s.save(&i, tx))));
        let i = id.clone();
        actions = actions.child(act(&icons::REVEAL, "reveal", "Reveal in Finder", !done, Box::new(move |s, _| s.reveal(&i))));
        let prompt = g.prompt.clone();
        actions = actions.child(act(&icons::COPY, "copy-prompt", "Copy prompt", false, Box::new(move |_, tx| tx.copy(prompt.clone()))));
        let seed = g.seed.to_string();
        actions = actions.child(act(&icons::HASH, "copy-seed", "Copy seed", false, Box::new(move |_, tx| tx.copy(seed.clone()))));
        let i = id.clone();
        actions = actions.child(act(&icons::RERUN, "rerun", "Re-run with a new seed", !s.connected(), Box::new(move |s, _| s.rerun(&i))));
        let i = id.clone();
        actions = actions.child(act(&icons::TRASH, "delete", "Delete", running, Box::new(move |s, _| s.delete(&i))));
    }
    actions = actions.child(
        IconButton::new(&shell::INSPECTOR)
            .variant(IconButtonVariant::GhostMuted)
            .control_size(ControlSize::Large)
            .expanded(s.inspector.is_open())
            .label(if s.inspector.is_open() { "Hide composer" } else { "Show composer" })
            .on_press(|s: &mut State| s.inspector.toggle())
            .build(k)
            .key("inspector-toggle"),
    );
    let title = text(title).key("toolbar-title").role("heading").font(k.ty.ui_medium).color(k.c.text_base).truncate().shrink().min_w(0.0);
    Toolbar::new().leading(title).trailing(actions)
}

fn menu(s: &State, k: &Kit, trigger: El, layer: &mut Vec<El>) -> El {
    let Some((g, at)) = s.menu.as_ref().and_then(|(id, at)| Some((s.item(id)?, *at))) else { return trigger };
    let id = g.id.as_str();
    let done = g.file().is_some();
    let running = matches!(g.phase, Phase::Running { .. } | Phase::Sending);
    let item = |label: &str| rune_kit::MenuItem::new(label);
    let (a, b, c, d, e, f) = (id.to_owned(), id.to_owned(), g.prompt.clone(), g.seed.to_string(), id.to_owned(), id.to_owned());
    context_menu("gallery-menu", trigger, Some(at))
        .item(item("Save PNG…").icon(&icons::SAVE).disabled(!done).on_select_tx(move |s: &mut State, tx| {
            s.menu = None;
            s.save(&a, tx);
        }))
        .item(item("Reveal in Finder").icon(&icons::REVEAL).disabled(!done).on_select(move |s: &mut State| {
            s.menu = None;
            s.reveal(&b);
        }))
        .separator()
        .item(item("Copy prompt").icon(&icons::COPY).on_select_tx(move |s: &mut State, tx| {
            s.menu = None;
            tx.copy(c.clone());
        }))
        .item(item("Copy seed").icon(&icons::HASH).on_select_tx(move |s: &mut State, tx| {
            s.menu = None;
            tx.copy(d.clone());
        }))
        .item(item("Re-run with a new seed").icon(&icons::RERUN).disabled(!s.connected()).on_select(move |s: &mut State| {
            s.menu = None;
            s.rerun(&e);
        }))
        .separator()
        .item(item("Delete").icon(&icons::TRASH).danger(true).disabled(running).on_select(move |s: &mut State| {
            s.menu = None;
            s.delete(&f);
        }))
        .on_close(|s: &mut State| s.menu = None)
        .build(k, layer)
}

fn confirm_delete(s: &State, k: &Kit, layer: &mut Vec<El>) {
    let Some(g) = s.deleting.as_deref().and_then(|id| s.item(id)) else { return };
    let what = if g.prompt.is_empty() { "this image".to_owned() } else { format!("\u{201c}{}\u{201d}", g.prompt.lines().next().unwrap_or_default()) };
    let actions = row()
        .key("delete-actions")
        .w_full()
        .padding(16.0)
        .gap(8.0)
        .justify_end()
        .child(Button::new("Cancel").variant(ButtonVariant::Ghost).on_press(|s: &mut State| s.deleting = None).build(k).key("delete-cancel"))
        .child(Button::new("Delete").variant(ButtonVariant::Danger).on_press(|s: &mut State| s.confirm_delete()).build(k).key("delete-confirm"));
    let body = text(format!("This removes {what} from mmh3 server and from this Mac.")).key("delete-what").font(k.ty.ui).color(k.c.text_muted).wrap();
    Dialog::new("delete", "Delete this image?")
        .description("It can't be undone.")
        .width(400.0)
        .fit()
        .open(true)
        .on_close(|s: &mut State| s.deleting = None)
        .child(col().w_full().px(16.0).child(body))
        .child(actions)
        .build(k, layer);
}

fn notes(s: &State, k: &Kit) -> El {
    let mut region = ToastRegion::new();
    for n in &s.notes {
        let id = n.id;
        let mark = el().size(8.0, 8.0).radius(4.0).fill(if n.danger { k.c.danger.icon } else { k.c.success.icon });
        let mut t = Toast::new(format!("note-{id}"), n.title.clone()).lead(mark).closable().on_close(move |s: &mut State| s.notes.retain(|n| n.id != id));
        if let Some(d) = &n.detail {
            t = t.description(d.clone());
        }
        region = region.toast(t.build(k));
    }
    region.build(k)
}

pub fn model_name(m: &str) -> String {
    match m {
        "text_encoder" => "text encoder".into(),
        "dit" => "DiT".into(),
        "image_vae" => "image VAE".into(),
        "video_vae" => "video VAE".into(),
        "audio_vae" => "audio VAE".into(),
        other => other.replace('_', " "),
    }
}

pub fn gigabytes(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / 1e9)
}

fn status_bar(s: &State, k: &Kit) -> El {
    let c = &k.c;
    let (dot, state) = match &s.conn {
        Conn::Connecting => (c.warning.icon, format!("Connecting to {}…", s.host())),
        Conn::Connected(_) => (c.success.icon, format!("Connected to {}", s.host())),
        Conn::Unreadable(_) => (c.warning.icon, format!("Connected to {}, but its status couldn't be read", s.host())),
        Conn::Disconnected(_) => (c.danger.icon, format!("Not connected to {}", s.host())),
    };
    let item = |key: &str, words: String, ink: Color| text(words).key(key.to_owned()).font(k.ty.small).color(ink).truncate().shrink();
    let sep = || el().size(1.0, 12.0).fill(c.border_base).a11y_hidden();
    let mut bar = row()
        .key("status-bar")
        .role("status")
        .label("Server status")
        .w_full()
        .h(28.0)
        .px(14.0)
        .gap(10.0)
        .center_y()
        .border_sides([0.0, 1.0, 0.0, 0.0], c.border_muted)
        .fill(c.bg_base)
        .child(el().key("status-dot").size(8.0, 8.0).radius(4.0).fill(dot).a11y_hidden())
        .child(item("status-connection", state, c.text_muted));
    if let Conn::Connected(st) = &s.conn {
        let models = match &st.models {
            Some(m) if m.is_empty() => "No models resident".to_owned(),
            Some(m) => format!("Resident: {}", m.iter().map(|m| model_name(m)).collect::<Vec<_>>().join(", ")),
            None => "Models busy with a generation".to_owned(),
        };
        let queue = match (st.generating.is_some(), st.queued) {
            (false, 0) => "Queue idle".to_owned(),
            (true, 0) => "Generating, nothing waiting".to_owned(),
            (true, n) => format!("Generating, {n} waiting"),
            (false, n) => format!("{n} waiting"),
        };
        bar = bar.child(sep()).child(item("status-models", models, c.text_faint));
        if let Some(m) = st.memory {
            bar = bar.child(sep()).child(item("status-memory", format!("Memory {} free of {}", gigabytes(m.free), gigabytes(m.total)), c.text_faint));
        }
        bar = bar.child(sep()).child(item("status-queue", queue, c.text_faint));
    }
    if let Conn::Unreadable(why) = &s.conn {
        bar = bar.child(sep()).child(item("status-error", why.clone(), c.text_faint));
    }
    bar
}

use std::borrow::Cow;

use rune::{DragItem, El, SurfaceFit, col, el, memo, row, text};
use rune_kit::shell::Sidebar;
use rune_kit::{Kit, Progress, Spinner};

use crate::state::{Gen, Phase, Picture, State};
use crate::view::kit;

pub const ROW: f32 = 72.0;
const THUMB_PT: f32 = 56.0;

pub fn sidebar(s: &State, k: &Kit, dark: bool) -> El {
    let count = s.gens.len();
    let header = row()
        .key("gallery-header")
        .w_full()
        .h(28.0)
        .px(8.0)
        .center_y()
        .justify_between()
        .child(text("Generations").font(k.ty.small_medium).color(k.c.text_faint))
        .child(text(if count == 0 { String::new() } else { count.to_string() }).key("gallery-count").font(k.ty.small).color(k.c.text_faint));
    let sb = Sidebar::new().label("Generations").inset(0.0).header_padding(0.0, 6.0).row_gap(0.0).header(header);
    if count == 0 {
        let empty = col()
            .key("gallery-empty")
            .w_full()
            .padding(12.0)
            .gap(4.0)
            .child(text("Nothing yet").font(k.ty.ui_medium).color(k.c.text_muted))
            .child(text("Finished images and the queue show up here.").font(k.ty.small).color(k.c.text_faint).wrap());
        return sb.child(empty).build(k);
    }
    sb.row_height(ROW).virtual_rows(count, |s: &State, i| Cow::Borrowed(s.gens[i].id.as_str()), move |s: &State, i| gallery_row(s, &s.gens[i], dark)).build(k)
}

fn gallery_row(s: &State, g: &Gen, dark: bool) -> El {
    let selected = s.selected.as_deref() == Some(g.id.as_str());
    let tick = if matches!(g.phase, Phase::Running { .. }) { s.tick } else { 0 };
    let clock = s.clock;
    memo(format!("row-{}", g.id), (g.revision, selected, tick, dark), || build_row(g, selected, clock, kit(dark)))
}

pub fn elapsed(since: f64, clock: f64) -> String {
    format!("{:.1} s", (clock - since).max(0.0))
}

fn status_line(g: &Gen, clock: f64, k: &Kit) -> (String, rune::Color) {
    let c = &k.c;
    match &g.phase {
        Phase::Sending => ("Sending…".into(), c.text_faint),
        Phase::Queued => ("Queued".into(), c.text_faint),
        Phase::Running { step, steps, since } => (format!("Step {step} of {steps} · {}", elapsed(*since, clock)), c.text_muted),
        Phase::Done { seconds, .. } => (format!("{}×{} · {seconds:.1} s", g.width, g.height), c.text_faint),
        Phase::Failed(_) => ("Failed".into(), c.danger.fg),
    }
}

fn thumb(g: &Gen, k: &Kit) -> El {
    let c = &k.c;
    let mut t = el().key("thumb").size(THUMB_PT, THUMB_PT).radius(6.0).fill(c.layer_02).center().border(0.5, c.border_muted);
    t = match (&g.phase, &g.picture) {
        (_, Picture::Ready { thumb: Some(img), .. }) => {
            t.child(el().key("thumb-image").size(THUMB_PT, THUMB_PT).radius(6.0).pixels_fit(img.pixels(), SurfaceFit::Fill))
        }
        (Phase::Running { .. } | Phase::Sending | Phase::Queued, _) | (_, Picture::Loading) => t.child(Spinner::new().size(14.0).color(c.icon_faint).build(k)),
        (Phase::Failed(_), _) => t.child(text("!").font(k.ty.ui_medium).color(c.danger.fg)),
        _ => t,
    };
    if let Some(file) = g.file().cloned() {
        t = t.drag_out(move |_: &State| Some(DragItem::new().file(file.clone())));
    }
    t.a11y_hidden()
}

fn build_row(g: &Gen, selected: bool, clock: f64, k: &Kit) -> El {
    let c = &k.c;
    let (status, ink) = status_line(g, clock, k);
    let title = if g.prompt.is_empty() { "Untitled".to_owned() } else { g.prompt.replace('\n', " ") };
    let mut lines = col()
        .key("lines")
        .grow(1.0)
        .shrink()
        .min_w(0.0)
        .gap(3.0)
        .child(
            text(title.clone())
                .key("row-prompt")
                .font(k.ty.ui_compact)
                .color(if selected { c.text_base } else { c.text_muted })
                .max_lines(2)
                .wrap()
                .a11y_hidden(),
        )
        .child(text(status.clone()).key("row-status").font(k.ty.small).color(ink).truncate().a11y_hidden());
    if let Phase::Running { step, steps, .. } = g.phase {
        lines = lines.child(Progress::new(Some(step as f32 / steps.max(1) as f32)).build(k).key("row-progress").a11y_hidden());
    }
    let id = g.id.clone();
    let menu_id = g.id.clone();
    let mut r = row()
        .key(format!("gen-{}", g.id))
        .role("listitem")
        .label(title)
        .description(status)
        .w_full()
        .h(ROW)
        .px(8.0)
        .gap(10.0)
        .center_y()
        .radius(k.radius.md)
        .focusable(true)
        .focus_selected(selected)
        .hover_group()
        .on_press(move |s: &mut State| s.select(&id))
        .on_secondary_press(move |s: &mut State, p, _| {
            s.select(&menu_id);
            s.menu = Some((menu_id.clone(), p));
        })
        .child(thumb(g, k))
        .child(lines);
    r = k.interactive(r);
    if selected {
        r = r.fill(c.pressed);
    }
    r
}

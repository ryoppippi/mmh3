use std::sync::Arc;
use std::time::Duration;

use rune::{AppSpec, Chrome, ImageCache, KeyEvent, Tx};

mod composer;
mod gallery;
mod icons;
mod net;
mod preview;
mod state;
mod view;

use state::{Phase, State};

fn poll(s: &mut State, now: f64, tx: &mut Tx) -> bool {
    s.clock = now;
    for n in &mut s.notes {
        n.until.get_or_insert(now + state::NOTE_SECONDS);
    }
    let before = s.notes.len();
    s.notes.retain(|n| n.until.is_some_and(|until| until > now));
    let mut changed = s.notes.len() != before;
    if let Some(next) = s.notes.iter().filter_map(|n| n.until).reduce(f64::min) {
        tx.wake_at(next);
    }
    if s.gens.iter().any(|g| matches!(g.phase, Phase::Running { .. })) {
        tx.wake_after(Duration::from_millis(100));
        let tick = (now * 10.0) as u64;
        changed |= tick != s.tick;
        s.tick = tick;
    }
    changed
}

fn on_key(s: &mut State, ev: &KeyEvent, _: &mut Tx) -> bool {
    if composer::is_generate(ev) {
        s.generate();
        return true;
    }
    if rune_kit::shell::is_sidebar_toggle(ev) {
        s.sidebar.toggle();
        return true;
    }
    if ev.is_command('i') && ev.mods.alt {
        s.inspector.toggle();
        return true;
    }
    false
}

fn spec() -> AppSpec<State> {
    AppSpec {
        min_size: Some((1040.0, 680.0)),
        poll: Some(poll),
        on_key: Some(on_key),
        ..AppSpec::new("mmh3 studio", (1320.0, 860.0), State::offline, view::view)
    }
}

fn main() {
    rune::prewarm_gpu();
    let app = rune::App::new(spec());
    let handle = app.handle();
    let ready = handle.clone();
    let chrome = Chrome { full_size_content: true, traffic_lights: Some((16.0, 25.0)), drag_regions: true };
    app.chrome(chrome)
        .init(move || {
            let waker = handle.clone();
            let images = Arc::new(ImageCache::new(Arc::new(move || waker.wake())));
            let server = state::server_from_args();
            let net = net::Net::new(handle, &server, images.clone());
            State::new(Some(net), images, server)
        })
        .on_ready(move |_| {
            ready.update(|s, _| {
                if let Some(net) = &s.net {
                    net.start();
                }
            });
        })
        .run();
}

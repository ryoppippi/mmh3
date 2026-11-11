//! Fixture states for headless snapshots (`--scene NAME`) and tests.

use std::path::PathBuf;

use mmh3_studio::api::{Generation, Kind, Memory, State as Remote, Status, Timings};
use mmh3_studio::mock::paint;

use crate::state::{Conn, Gen, Phase, Picture, SizeChoice, State};

type Scene = (&'static str, fn(&mut State));

pub const SCENES: &[Scene] = &[
    ("empty", empty),
    ("disconnected", disconnected),
    ("running", running),
    ("gallery", gallery),
    ("failed", failed),
    ("custom-size", custom_size),
    ("offline-selected", offline_selected),
    ("confirm-delete", confirm_delete),
];

pub fn status(busy: bool, queued: u32) -> Status {
    Status {
        generating: busy.then(|| "7c01aa2e0005".to_owned()),
        generations: 5,
        memory: Some(Memory { free: if busy { 61_400_000_000 } else { 84_100_000_000 }, total: 137_400_000_000 }),
        models: (!busy).then(|| vec!["text_encoder".into(), "dit".into(), "image_vae".into()]),
        queued,
    }
}

/// Scenes are fixtures: they never reach a server, whatever the run was started with.
fn offline(s: &mut State) {
    s.net = None;
}

pub fn empty(s: &mut State) {
    offline(s);
    s.conn = Conn::Connected(status(false, 0));
}

pub fn disconnected(s: &mut State) {
    offline(s);
    s.conn = Conn::Disconnected("Connection refused (os error 61)".into());
}

fn remote(id: &str, prompt: &str, seed: u64, (width, height): (u32, u32), state: Remote) -> Generation {
    Generation {
        id: id.into(),
        state,
        queued: 1_794_393_000,
        seconds: None,
        step: None,
        steps: None,
        error: None,
        kind: Kind::Image,
        width: Some(width),
        height: Some(height),
        prompt: Some(prompt.into()),
        seed: Some(seed),
        timings: None,
    }
}

fn done(id: &str, prompt: &str, seed: u64, size: (u32, u32)) -> Generation {
    let steps = vec![1.98, 1.97, 1.99, 1.97, 1.98, 1.96];
    Generation { seconds: Some(14.62), timings: Some(Timings { text: 0.46, steps, decode: 2.31 }), ..remote(id, prompt, seed, size, Remote::Done) }
}

/// Paints a fixture at a fraction of the generation's size, with a decoded thumbnail.
fn picture(s: &mut State, id: &str, preview: bool) {
    let Some(g) = s.gens.iter().find(|g| g.id == id) else { return };
    let (w, h, seed) = (g.width / 4, g.height / 4, g.seed);
    let png = paint(w - w % 8, h - h % 8, seed);
    let thumb = mmh3_studio::thumb::thumbnail(&png).ok().map(|t| s.images.rgba(t.width(), t.height(), t.into_raw()));
    if let Some(t) = &thumb {
        let _ = t.wait();
    }
    if preview {
        let full = s.images.load(png);
        let _ = full.wait();
        s.preview = Some((id.to_owned(), full));
    }
    if let Some(g) = s.gens.iter_mut().find(|g| g.id == id) {
        g.picture = Picture::Ready { file: PathBuf::from(format!("/tmp/mmh3-studio-fixture-{id}.png")), thumb };
    }
}

fn library(s: &mut State) {
    offline(s);
    let list = vec![
        done("7c01aa2e0004", "Paper boats drifting across a rain puddle on cobblestones, reflections of neon signs", 912, (4736, 3552)),
        done("7c01aa2e0003", "A lighthouse on a basalt cliff at dusk, long exposure, sea mist", 4127, (5472, 3072)),
        done("7c01aa2e0002", "Isometric illustration of a tiny greenhouse full of succulents", 77, (2048, 2048)),
        done("7c01aa2e0001", "Portrait of a red fox in fresh snow, shallow depth of field", 31_337, (3072, 5472)),
    ];
    s.merge(list, 0.0);
    for id in ["7c01aa2e0004", "7c01aa2e0003", "7c01aa2e0002", "7c01aa2e0001"] {
        picture(s, id, false);
    }
}

pub fn running(s: &mut State) {
    s.conn = Conn::Connected(status(true, 1));
    library(s);
    let mut running = remote("7c01aa2e0005", "A cathedral made of glass in a pine forest, morning fog, volumetric light", 2026, (4096, 4096), Remote::Running);
    running.step = Some(3);
    running.steps = Some(6);
    let queued = remote("7c01aa2e0006", "Macro photograph of frost ferns on a window pane", 8, (2048, 2048), Remote::Queued);
    let mut gens = vec![Gen::from_server(&queued, 0.0), Gen::from_server(&running, 0.0)];
    if let Phase::Running { since, .. } = &mut gens[1].phase {
        *since = -7.4;
    }
    gens.append(&mut s.gens);
    s.gens = gens;
    s.prompt.replace_text("A cathedral made of glass in a pine forest, morning fog, volumetric light");
    s.size = SizeChoice::Preset(1);
    s.selected = Some("7c01aa2e0005".into());
}

pub fn gallery(s: &mut State) {
    s.conn = Conn::Connected(status(false, 0));
    library(s);
    s.selected = Some("7c01aa2e0003".into());
    picture(s, "7c01aa2e0003", true);
}

pub fn failed(s: &mut State) {
    s.conn = Conn::Connected(status(false, 0));
    library(s);
    let mut f = remote("7c01aa2e0007", "A city skyline made of stacked books", 5, (2048, 2048), Remote::Failed);
    f.error = Some("The DiT ran out of memory at step 4 (requested 23.1 GB, 19.8 GB free).".into());
    s.gens.insert(0, Gen::from_server(&f, 0.0));
    s.selected = Some("7c01aa2e0007".into());
}

pub fn custom_size(s: &mut State) {
    empty(s);
    s.prompt.replace_text("Wide panorama of the Dolomites at sunrise");
    s.size = SizeChoice::Custom;
    s.custom_w.replace_text("6000");
    s.custom_h.replace_text("2000");
}

pub fn offline_selected(s: &mut State) {
    gallery(s);
    disconnected(s);
}

pub fn confirm_delete(s: &mut State) {
    gallery(s);
    s.delete("7c01aa2e0003");
}

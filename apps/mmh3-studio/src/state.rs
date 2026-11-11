use std::collections::HashSet;
use std::hash::{BuildHasher, Hasher};
use std::path::PathBuf;
use std::sync::Arc;

use mmh3_studio::api::{self, Generation, ImageRequest, Kind, Status, Timings};
use mmh3_studio::sizes::{self, PRESETS, SizeError};
use rune::machine::Effects;
use rune::{Image, ImageCache, PointerPress, TextEdit, Tx};
use rune_kit::image_zoom::StageZoom;
use rune_kit::shell::PaneState;

use crate::net::Net;

/// Thumbnails kept decoded, newest first; older rows show a placeholder.
pub const MAX_THUMBS: usize = 240;
pub const NOTE_SECONDS: f64 = 4.0;

pub enum Conn {
    Connecting,
    Connected(Status),
    /// The server answered, but not in a way the studio could read.
    Unreadable(String),
    Disconnected(String),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Phase {
    /// Posted, waiting for the server's 202.
    Sending,
    Queued,
    Running {
        step: u32,
        steps: u32,
        since: f64,
    },
    Done {
        seconds: f64,
        timings: Option<Timings>,
    },
    Failed(String),
}

impl Phase {
    pub fn is_active(&self) -> bool {
        matches!(self, Phase::Sending | Phase::Queued | Phase::Running { .. })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Effect {
    Watch,
    FetchImage,
}

/// Feeds what the server says about a generation into its phase.
pub fn step(phase: &mut Phase, g: &Generation, now: f64) -> Effects<Effect> {
    let was = phase.clone();
    *phase = match g.state {
        api::State::Queued => Phase::Queued,
        api::State::Running => {
            let since = match was {
                Phase::Running { since, .. } => since,
                _ => now,
            };
            Phase::Running { step: g.step.unwrap_or(0), steps: g.steps.unwrap_or(api::IMAGE_STEPS), since }
        }
        api::State::Done => Phase::Done { seconds: g.seconds.unwrap_or_default(), timings: g.timings.clone() },
        api::State::Failed => Phase::Failed(g.error.clone().unwrap_or_else(|| "The server gave no reason.".into())),
    };
    match (&was, &*phase) {
        (Phase::Done { .. }, _) => Effects::none(),
        (_, Phase::Done { .. }) => Effects::one(Effect::FetchImage),
        (Phase::Sending, p) if p.is_active() => Effects::one(Effect::Watch),
        _ => Effects::none(),
    }
}

pub enum Picture {
    None,
    Loading,
    Ready { file: PathBuf, thumb: Option<Image> },
    Failed(String),
}

pub struct Gen {
    pub id: String,
    pub prompt: String,
    pub seed: u64,
    pub width: u32,
    pub height: u32,
    pub phase: Phase,
    pub picture: Picture,
    pub revision: u64,
}

impl Gen {
    pub fn from_server(g: &Generation, now: f64) -> Gen {
        let mut item = Gen {
            id: g.id.clone(),
            prompt: g.prompt.clone().unwrap_or_default(),
            seed: g.seed.unwrap_or_default(),
            width: g.width.unwrap_or_default(),
            height: g.height.unwrap_or_default(),
            phase: Phase::Queued,
            picture: Picture::None,
            revision: 0,
        };
        let _ = step(&mut item.phase, g, now);
        item
    }

    pub fn file(&self) -> Option<&PathBuf> {
        match &self.picture {
            Picture::Ready { file, .. } => Some(file),
            _ => None,
        }
    }

    pub fn is_local(&self) -> bool {
        self.id.starts_with("local-")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SizeChoice {
    Preset(usize),
    Custom,
}

pub struct Note {
    pub id: u64,
    pub title: String,
    pub detail: Option<String>,
    pub danger: bool,
    /// Stamped from the frame clock by the first poll that sees the note, so it stays up its
    /// full time however long the window sat idle before it arrived.
    pub until: Option<f64>,
}

pub struct State {
    pub net: Option<Net>,
    pub images: Arc<ImageCache>,
    pub server: String,
    pub conn: Conn,
    pub gens: Vec<Gen>,
    pub selected: Option<String>,
    pub prompt: TextEdit,
    pub size: SizeChoice,
    pub custom_w: TextEdit,
    pub custom_h: TextEdit,
    pub seed: TextEdit,
    pub preview: Option<(String, Image)>,
    pub stage: StageZoom,
    pub sidebar: PaneState,
    pub inspector: PaneState,
    pub menu: Option<(String, PointerPress)>,
    pub deleting: Option<String>,
    pub notes: Vec<Note>,
    pub clock: f64,
    pub tick: u64,
    pub watching: HashSet<String>,
    next_local: u64,
}

pub fn random_seed() -> u64 {
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos()));
    h.finish() % 4_294_967_296
}

pub fn server_from_args() -> String {
    let args = rune::extra_args();
    let flag = args.iter().position(|a| a == "--server").and_then(|i| args.get(i + 1)).cloned();
    flag.or_else(|| std::env::var("MMH3_SERVER").ok()).unwrap_or_else(|| api::DEFAULT_SERVER.to_owned())
}

impl State {
    pub fn new(net: Option<Net>, images: Arc<ImageCache>, server: String) -> State {
        State {
            net,
            images,
            server,
            conn: Conn::Connecting,
            gens: Vec::new(),
            selected: None,
            prompt: TextEdit::new(""),
            size: SizeChoice::Preset(0),
            custom_w: TextEdit::new("1536"),
            custom_h: TextEdit::new("1024"),
            seed: TextEdit::new(random_seed().to_string()),
            preview: None,
            stage: StageZoom::default(),
            sidebar: PaneState::new(300.0).range(220.0, 480.0),
            inspector: PaneState::new(340.0).range(300.0, 460.0),
            menu: None,
            deleting: None,
            notes: Vec::new(),
            clock: 0.0,
            tick: 0,
            watching: HashSet::new(),
            next_local: 0,
        }
    }

    pub fn offline() -> State {
        State::new(None, Arc::new(ImageCache::new(Arc::new(|| {}))), api::DEFAULT_SERVER.to_owned())
    }

    pub fn item(&self, id: &str) -> Option<&Gen> {
        self.gens.iter().find(|g| g.id == id)
    }

    fn item_mut(&mut self, id: &str) -> Option<&mut Gen> {
        self.gens.iter_mut().find(|g| g.id == id)
    }

    pub fn selected_gen(&self) -> Option<&Gen> {
        self.selected.as_deref().and_then(|id| self.item(id))
    }

    pub fn connected(&self) -> bool {
        matches!(self.conn, Conn::Connected(_) | Conn::Unreadable(_))
    }

    fn reachable(&self) -> Result<(), String> {
        if self.connected() { Ok(()) } else { Err("Start mmh3 server first.".into()) }
    }

    pub fn active_count(&self) -> usize {
        self.gens.iter().filter(|g| g.phase.is_active()).count()
    }

    pub fn host(&self) -> &str {
        self.server.split("://").nth(1).unwrap_or(&self.server).trim_end_matches('/')
    }

    pub fn chosen_size(&self) -> Result<(u32, u32), SizeError> {
        match self.size {
            SizeChoice::Preset(i) => Ok((PRESETS[i].width, PRESETS[i].height)),
            SizeChoice::Custom => sizes::custom(&self.custom_w.text, &self.custom_h.text),
        }
    }

    pub fn seed_value(&self) -> Option<u64> {
        let t = self.seed.text.trim();
        if t.is_empty() { Some(random_seed()) } else { t.parse().ok() }
    }

    /// What Generate would send, or why it can't: the one rule the button, its help and the
    /// keyboard shortcut share.
    pub fn request(&self) -> Result<ImageRequest, String> {
        self.reachable()?;
        let prompt = self.prompt.text.trim();
        if prompt.is_empty() {
            return Err("Write a prompt first.".into());
        }
        let (width, height) = self.chosen_size().map_err(|e| e.to_string())?;
        let seed = self.seed_value().ok_or("The seed must be a whole number.")?;
        Ok(ImageRequest { prompt: prompt.to_owned(), seed, width, height, min_generation_area: None })
    }

    pub fn generate(&mut self) {
        match self.request() {
            Ok(r) => self.submit(r),
            Err(why) => self.note(why, None, true),
        }
    }

    pub fn rerun(&mut self, id: &str) {
        if let Err(why) = self.reachable() {
            return self.note(why, None, true);
        }
        let Some(g) = self.item(id) else { return };
        let r = ImageRequest { prompt: g.prompt.clone(), seed: random_seed(), width: g.width, height: g.height, min_generation_area: None };
        self.submit(r);
    }

    pub fn submit(&mut self, r: ImageRequest) {
        self.next_local += 1;
        let id = format!("local-{}", self.next_local);
        self.gens.insert(
            0,
            Gen {
                id: id.clone(),
                prompt: r.prompt.clone(),
                seed: r.seed,
                width: r.width,
                height: r.height,
                phase: Phase::Sending,
                picture: Picture::None,
                revision: 0,
            },
        );
        self.select(&id);
        if let Some(net) = &self.net {
            net.submit(id, r);
        }
    }

    /// The server's answer to a submission: the local placeholder takes the server's id, or
    /// gives way to the row a list refresh already made for it.
    pub fn submitted(&mut self, local: &str, answer: Result<Generation, api::Error>, now: f64) {
        let created = match answer {
            Ok(created) => created,
            Err(e) => {
                if let Some(g) = self.item_mut(local) {
                    g.phase = Phase::Failed(e.to_string());
                    g.revision += 1;
                }
                return;
            }
        };
        let was_selected = self.selected.as_deref() == Some(local);
        if self.item(&created.id).is_some() {
            self.gens.retain(|g| g.id != local);
        } else {
            let Some(g) = self.item_mut(local) else { return };
            g.id = created.id.clone();
            g.revision += 1;
            self.apply(&created, now);
        }
        if was_selected {
            self.selected = None;
            self.select(&created.id);
        }
    }

    pub fn apply(&mut self, g: &Generation, now: f64) {
        let Some(item) = self.item_mut(&g.id) else { return };
        let effects = step(&mut item.phase, g, now);
        item.revision += 1;
        for e in effects {
            self.run(&g.id, e);
        }
    }

    fn run(&mut self, id: &str, effect: Effect) {
        match effect {
            Effect::Watch => self.watch(id),
            Effect::FetchImage => self.fetch_image(id, true),
        }
    }

    pub fn watch(&mut self, id: &str) {
        let Some(net) = &self.net else { return };
        if self.watching.insert(id.to_owned()) {
            net.watch(id.to_owned());
        }
    }

    /// Asks for a finished generation's picture: `urgent` for one the user waits for (just
    /// finished, or selected), which goes ahead of the gallery's backlog.
    pub fn fetch_image(&mut self, id: &str, urgent: bool) {
        let Some(net) = &self.net else { return };
        let Some(g) = self.gens.iter_mut().find(|g| g.id == id) else { return };
        if !matches!(g.phase, Phase::Done { .. }) {
            return;
        }
        match g.picture {
            Picture::None | Picture::Failed(_) => {
                g.picture = Picture::Loading;
                g.revision += 1;
                net.fetch_image(id, urgent);
            }
            Picture::Loading if urgent => net.fetch_image(id, true),
            _ => {}
        }
    }

    pub fn watch_ended(&mut self, id: &str) {
        self.watching.remove(id);
    }

    /// The server's list, most recent first: keeps what the app already holds for each, adds
    /// the new ones and drops what the server no longer has. Pictures are fetched for the rows
    /// that keep a thumbnail; older ones wait until they are selected.
    pub fn merge(&mut self, list: Vec<Generation>, now: f64) {
        let mut old: Vec<Gen> = std::mem::take(&mut self.gens);
        let mut gens: Vec<Gen> = Vec::new();
        let mut i = 0;
        while i < old.len() {
            if old[i].is_local() { gens.push(old.remove(i)) } else { i += 1 }
        }
        let mut effects = Vec::new();
        for g in list.iter().filter(|g| g.kind == Kind::Image) {
            let item = match old.iter().position(|o| o.id == g.id) {
                Some(at) => {
                    let mut item = old.remove(at);
                    let fx = step(&mut item.phase, g, now);
                    item.revision += 1;
                    effects.extend(fx.into_iter().map(|e| (g.id.clone(), e)));
                    item
                }
                None => Gen::from_server(g, now),
            };
            if item.phase.is_active() {
                effects.push((g.id.clone(), Effect::Watch));
            }
            gens.push(item);
        }
        self.gens = gens;
        let gone: Vec<String> = old.into_iter().filter(|o| !o.is_local()).map(|o| o.id).collect();
        if let (Some(net), false) = (&self.net, gone.is_empty()) {
            net.forget(&gone);
        }
        let backlog: Vec<String> = self.gens.iter().filter(|g| matches!(g.phase, Phase::Done { .. })).take(MAX_THUMBS).map(|g| g.id.clone()).collect();
        if let Some(sel) = &self.selected
            && self.item(sel).is_none()
        {
            self.selected = None;
            self.preview = None;
        }
        for (id, e) in effects {
            self.run(&id, e);
        }
        if let Some(sel) = self.selected.clone() {
            self.fetch_image(&sel, true);
        }
        for id in backlog {
            self.fetch_image(&id, false);
        }
        self.trim_thumbs();
    }

    pub fn picture_ready(&mut self, id: &str, file: PathBuf, thumb: Image) {
        let Some(g) = self.item_mut(id) else { return };
        g.picture = Picture::Ready { file, thumb: Some(thumb) };
        g.revision += 1;
        if self.selected.as_deref() == Some(id) {
            self.load_preview();
        }
        self.trim_thumbs();
    }

    pub fn picture_failed(&mut self, id: &str, why: String) {
        if let Some(g) = self.item_mut(id) {
            g.picture = Picture::Failed(why);
            g.revision += 1;
        }
    }

    fn trim_thumbs(&mut self) {
        let mut kept = 0;
        for g in &mut self.gens {
            if let Picture::Ready { thumb, .. } = &mut g.picture
                && thumb.is_some()
            {
                kept += 1;
                if kept > MAX_THUMBS {
                    *thumb = None;
                    g.revision += 1;
                }
            }
        }
    }

    pub fn select(&mut self, id: &str) {
        if self.selected.as_deref() == Some(id) {
            return;
        }
        self.selected = Some(id.to_owned());
        self.stage = StageZoom::default();
        self.fetch_image(id, true);
        self.load_preview();
    }

    /// The selected image at full size, decoded on the image cache's workers. Only one is held,
    /// so a 16 MP image's pixels are freed when the selection moves on.
    fn load_preview(&mut self) {
        let wanted = self.selected_gen().and_then(|g| g.file().map(|f| (g.id.clone(), f.clone())));
        self.preview = match wanted {
            Some((id, _)) if self.preview.as_ref().is_some_and(|(p, _)| *p == id) => self.preview.take(),
            Some((id, file)) => Some((id, self.images.file(file, 0.0, 2.0))),
            None => None,
        };
    }

    pub fn deleted(&mut self, id: &str) {
        self.gens.retain(|g| g.id != id);
        if self.selected.as_deref() == Some(id) {
            self.selected = None;
            self.preview = None;
        }
        if self.deleting.as_deref() == Some(id) {
            self.deleting = None;
        }
    }

    /// Asks before deleting what the server holds; a placeholder that never reached it goes
    /// at once.
    pub fn delete(&mut self, id: &str) {
        let Some(g) = self.item(id) else { return };
        if matches!(g.phase, Phase::Running { .. } | Phase::Sending) {
            return self.note("A running generation can't be deleted.", Some("Wait for it to finish, then delete it.".into()), true);
        }
        if g.is_local() {
            return self.deleted(id);
        }
        self.deleting = Some(id.to_owned());
    }

    pub fn confirm_delete(&mut self) {
        let Some(id) = self.deleting.take() else { return };
        match &self.net {
            Some(net) => net.delete(id),
            None => self.deleted(&id),
        }
    }

    pub fn note(&mut self, title: impl Into<String>, detail: Option<String>, danger: bool) {
        let id = self.notes.iter().map(|n| n.id + 1).max().unwrap_or(1);
        self.notes.push(Note { id, title: title.into(), detail, danger, until: None });
        if self.notes.len() > 3 {
            self.notes.remove(0);
        }
    }

    pub fn set_status(&mut self, status: Result<Status, api::Error>) {
        self.conn = match status {
            Ok(s) => Conn::Connected(s),
            Err(e) if e.is_unreachable() => Conn::Disconnected(e.to_string()),
            Err(e) => Conn::Unreadable(e.to_string()),
        };
    }

    pub fn randomise_seed(&mut self) {
        self.seed.replace_text(random_seed().to_string());
    }

    pub fn save(&mut self, id: &str, tx: &mut Tx) {
        let Some(file) = self.item(id).and_then(|g| g.file().cloned()) else { return };
        let options = rune::FileDialogOptions { title: "Save image".into(), file_name: Some(format!("mmh3-{id}.png")), ..Default::default() }
            .filter("PNG image", &["png"]);
        tx.save_file(options, move |s: &mut State, answer, _tx| match answer {
            Ok(Some(to)) => match &s.net {
                Some(net) => net.save(file.clone(), to),
                None => s.note("Saving needs the server connection.", None, true),
            },
            Ok(None) => {}
            Err(e) => s.note("Couldn't open the save dialog.", Some(format!("{e:?}")), true),
        });
    }

    pub fn reveal(&mut self, id: &str) {
        if let Some(file) = self.item(id).and_then(|g| g.file().cloned()) {
            crate::net::reveal(file);
        }
    }
}

#[cfg(test)]
mod tests;

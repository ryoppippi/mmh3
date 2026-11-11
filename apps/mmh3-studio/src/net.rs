//! The studio's background work. Every request runs on its own thread and lands its answer in
//! the state through the app handle, which commits it on the next frame.

use std::collections::{HashSet, VecDeque};
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use mmh3_studio::api::{Client, Error, Generation, ImageRequest, Status};
use mmh3_studio::thumb::thumbnail;
use rune::{AppHandle, ImageCache};

use crate::state::State;

const STATUS_EVERY: Duration = Duration::from_secs(2);
const WATCH_RETRY: Duration = Duration::from_millis(500);
const WATCH_RETRY_MAX: Duration = Duration::from_secs(5);

/// The pictures still to fetch: what the user waits for first, then the gallery's backlog.
/// `wanted` holds every id queued or being fetched, so a forgotten one is never written.
#[derive(Default)]
struct Queue {
    urgent: VecDeque<String>,
    backlog: VecDeque<String>,
    wanted: HashSet<String>,
}

impl Queue {
    fn next(&mut self) -> Option<String> {
        self.urgent.pop_front().or_else(|| self.backlog.pop_front())
    }

    fn unqueue(&mut self, id: &str) -> bool {
        let before = self.urgent.len() + self.backlog.len();
        self.urgent.retain(|q| q != id);
        self.backlog.retain(|q| q != id);
        before != self.urgent.len() + self.backlog.len()
    }
}

struct Pictures {
    dir: PathBuf,
    queue: Mutex<Queue>,
}

impl Pictures {
    fn lock(&self) -> MutexGuard<'_, Queue> {
        self.queue.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn file(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.png"))
    }

    /// Drops `ids` from the queue and their cached files, under the lock a fetch writes under.
    fn forget<'a>(&self, ids: impl IntoIterator<Item = &'a str>) {
        let mut q = self.lock();
        for id in ids {
            q.unqueue(id);
            q.wanted.remove(id);
            let _ = std::fs::remove_file(self.file(id));
        }
    }

    /// Removes cached files of generations the server no longer lists.
    fn prune(&self, listed: &HashSet<String>) {
        let Ok(entries) = std::fs::read_dir(&self.dir) else { return };
        let q = self.lock();
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(id) = name.to_str().and_then(|n| n.strip_suffix(".png").or_else(|| n.strip_suffix(".png.part"))) else { continue };
            if !listed.contains(id) && !q.wanted.contains(id) {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }

    /// The generation's PNG, from the cache folder or the server, and its thumbnail; `None`
    /// when the generation was forgotten while it was being fetched.
    fn fetch(&self, client: &Client, id: &str) -> Result<Option<(PathBuf, image::RgbaImage)>, String> {
        let file = self.file(id);
        if let Ok(png) = std::fs::read(&file) {
            match thumbnail(&png) {
                Ok(thumb) => return Ok(Some((file, thumb))),
                Err(_) => {
                    let _ = std::fs::remove_file(&file);
                }
            }
        }
        let png = client.image(id).map_err(|e| e.to_string())?;
        std::fs::create_dir_all(&self.dir).map_err(|e| e.to_string())?;
        let part = self.dir.join(format!("{id}.png.part"));
        std::fs::write(&part, &png).map_err(|e| e.to_string())?;
        {
            let q = self.lock();
            if !q.wanted.contains(id) {
                let _ = std::fs::remove_file(&part);
                return Ok(None);
            }
            std::fs::rename(&part, &file).map_err(|e| e.to_string())?;
        }
        Ok(Some((file, thumbnail(&png)?)))
    }
}

pub struct Net {
    handle: AppHandle<State>,
    client: Client,
    pictures: Arc<Pictures>,
    wake: Sender<()>,
}

impl Net {
    pub fn new(handle: AppHandle<State>, server: &str, images: Arc<ImageCache>) -> Net {
        let client = Client::new(server);
        let root = rune::dirs::AppDirs::new("dev.mmh3.studio").map_or_else(|| std::env::temp_dir().join("mmh3-studio"), |d| d.cache().join("images"));
        let pictures = Arc::new(Pictures { dir: root.join(folder_for(client.base())), queue: Mutex::default() });
        let (wake, rx) = channel::<()>();
        let (h, c, p) = (handle.clone(), client.clone(), pictures.clone());
        let _ = rune::task::serve("mmh3-pictures", rx, move |()| {
            let Some(id) = p.lock().next() else { return ControlFlow::Continue(()) };
            let answer = p.fetch(&c, &id);
            p.lock().wanted.remove(&id);
            let answer = match answer {
                Ok(None) => return ControlFlow::Continue(()),
                Ok(Some((file, thumb))) => Ok((file, images.rgba(thumb.width(), thumb.height(), thumb.into_raw()))),
                Err(why) => Err(why),
            };
            h.update(move |s, _| match answer {
                Ok((file, thumb)) => s.picture_ready(&id, file, thumb),
                Err(why) => s.picture_failed(&id, why),
            });
            ControlFlow::Continue(())
        });
        Net { handle, client, pictures, wake }
    }

    /// Polls the server's status every two seconds, landing it only when it changes; a change
    /// in the number of generations, or a change in whether the server answers, refreshes the
    /// list.
    pub fn start(&self) {
        let (h, c, p) = (self.handle.clone(), self.client.clone(), self.pictures.clone());
        let _ = rune::task::spawn("mmh3-status", move || {
            let mut last: Option<Result<Status, Error>> = None;
            loop {
                let now = c.status();
                let refresh = match (&last, &now) {
                    (Some(Ok(a)), Ok(b)) => a.generations != b.generations,
                    (_, Err(e)) if e.is_unreachable() => false,
                    (Some(was), _) => *was != now,
                    (None, _) => true,
                };
                if last.as_ref() != Some(&now) {
                    let status = now.clone();
                    h.update(move |s, tx| {
                        tx.instant();
                        s.set_status(status);
                    });
                }
                if refresh && let Ok(list) = c.list() {
                    p.prune(&list.iter().map(|g| g.id.clone()).collect());
                    h.update(move |s, tx| {
                        tx.instant();
                        let now = tx.clock();
                        s.merge(list, now);
                    });
                }
                last = Some(now);
                rune::task::sleep(STATUS_EVERY);
            }
        });
    }

    pub fn submit(&self, local: String, request: ImageRequest) {
        let (h, c) = (self.handle.clone(), self.client.clone());
        let _ = rune::task::spawn("mmh3-submit", move || {
            let answer = c.create_image(&request);
            h.update(move |s, tx| {
                let now = tx.clock();
                s.submitted(&local, answer, now);
            });
        });
    }

    /// Follows a generation until it finishes. A stream that ends early is picked up again, and
    /// an unreachable server is retried, so the row never stays stuck at its last step; a
    /// generation the server no longer has is dropped.
    pub fn watch(&self, id: String) {
        let (h, c, p) = (self.handle.clone(), self.client.clone(), self.pictures.clone());
        let _ = rune::task::spawn("mmh3-events", move || {
            let land = |g: Generation| {
                h.update(move |s, tx| {
                    let now = tx.clock();
                    s.apply(&g, now);
                })
            };
            let mut wait = WATCH_RETRY;
            let lost = loop {
                let mut heard = false;
                let _ = c.events(&id, |g| {
                    heard = true;
                    land(g);
                });
                if heard {
                    wait = WATCH_RETRY;
                }
                match c.generation(&id) {
                    Ok(g) if g.is_finished() => {
                        land(g);
                        break None;
                    }
                    Ok(g) => land(g),
                    Err(Error::Refused { status: 404, .. }) => {
                        p.forget([id.as_str()]);
                        let gone = id.clone();
                        h.update(move |s, _| s.deleted(&gone));
                        break None;
                    }
                    Err(e) if e.is_unreachable() => {}
                    Err(e) => break Some(e),
                }
                rune::task::sleep(wait);
                wait = (wait * 2).min(WATCH_RETRY_MAX);
            };
            h.update(move |s, _| {
                s.watch_ended(&id);
                if let Some(e) = lost {
                    s.note("Lost track of a generation.", Some(e.to_string()), true);
                }
            });
        });
    }

    /// Queues the generation's picture, or moves it to the front when `urgent`.
    pub fn fetch_image(&self, id: &str, urgent: bool) {
        let mut q = self.pictures.lock();
        if q.wanted.contains(id) {
            if urgent && q.unqueue(id) {
                q.urgent.push_front(id.to_owned());
            }
            return;
        }
        q.wanted.insert(id.to_owned());
        if urgent {
            q.urgent.push_front(id.to_owned());
        } else {
            q.backlog.push_back(id.to_owned());
        }
        drop(q);
        let _ = self.wake.send(());
    }

    pub fn forget(&self, ids: &[String]) {
        self.pictures.forget(ids.iter().map(String::as_str));
    }

    pub fn delete(&self, id: String) {
        let (h, c, p) = (self.handle.clone(), self.client.clone(), self.pictures.clone());
        let _ = rune::task::spawn("mmh3-delete", move || {
            let answer = c.delete(&id);
            if answer.is_ok() {
                p.forget([id.as_str()]);
            }
            h.update(move |s, _| match answer {
                Ok(()) => s.deleted(&id),
                Err(e) => s.note("Couldn't delete the generation.", Some(e.to_string()), true),
            });
        });
    }

    pub fn save(&self, from: PathBuf, to: PathBuf) {
        let h = self.handle.clone();
        let _ = rune::task::spawn("mmh3-save", move || {
            let answer = std::fs::copy(&from, &to);
            h.update(move |s, _| match answer {
                Ok(_) => s.note("Image saved", Some(to.display().to_string()), false),
                Err(e) => s.note("Couldn't save the image.", Some(e.to_string()), true),
            });
        });
    }
}

/// One cache folder a server, since ids are only unique on the server that made them.
fn folder_for(base: &str) -> String {
    let host = base.split("://").nth(1).unwrap_or(base);
    host.chars().map(|c| if c.is_ascii_alphanumeric() || c == '.' { c } else { '-' }).collect()
}

pub fn reveal(file: PathBuf) {
    let _ = rune::task::spawn("mmh3-reveal", move || {
        let _ = std::process::Command::new("/usr/bin/open").arg("-R").arg(&file).status();
    });
}

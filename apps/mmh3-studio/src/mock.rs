//! A stand-in for `mmh3 server` in image mode: the same routes and JSON, six timed steps, and a
//! painted PNG of the requested size, so the studio can be exercised without the engine.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::api::{Generation, IMAGE_STEPS, Kind, Memory, State, Status, Timings};
use crate::sizes;

const MODELS: [&str; 3] = ["text_encoder", "dit", "image_vae"];
const FIELDS: [&str; 7] = ["image", "prompt", "seed", "width", "height", "min-generation-area", "steps"];
const TOTAL_MEMORY: u64 = 128 << 30;
const MAX_BODY: usize = 1 << 20;

#[derive(Clone, Debug)]
pub struct Config {
    pub text: Duration,
    pub step: Duration,
    pub decode: Duration,
}

impl Default for Config {
    fn default() -> Config {
        Config { text: Duration::from_millis(400), step: Duration::from_millis(600), decode: Duration::from_millis(700) }
    }
}

struct Job {
    generation: Generation,
    png: Option<Arc<Vec<u8>>>,
}

#[derive(Default)]
struct Jobs {
    list: Vec<Job>,
    next: u64,
}

struct Shared {
    jobs: Mutex<Jobs>,
    queued: Condvar,
    config: Config,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, Jobs> {
        self.jobs.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn update(&self, id: &str, f: impl FnOnce(&mut Job)) {
        if let Some(job) = self.lock().list.iter_mut().find(|j| j.generation.id == id) {
            f(job);
        }
    }
}

/// Binds `addr` and serves it on background threads; returns the address it bound.
pub fn spawn(addr: &str, config: Config) -> std::io::Result<SocketAddr> {
    let listener = TcpListener::bind(addr)?;
    let bound = listener.local_addr()?;
    let shared = Arc::new(Shared { jobs: Mutex::new(Jobs::default()), queued: Condvar::new(), config });
    let worker = shared.clone();
    std::thread::Builder::new().name("mock-worker".into()).spawn(move || work(&worker))?;
    std::thread::Builder::new().name("mock-accept".into()).spawn(move || {
        for stream in listener.incoming().flatten() {
            let shared = shared.clone();
            let _ = std::thread::Builder::new().name("mock-conn".into()).spawn(move || {
                let _ = handle(stream, &shared);
            });
        }
    })?;
    Ok(bound)
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

fn work(shared: &Shared) {
    loop {
        let (id, prompt, seed, width, height) = {
            let mut jobs = shared.lock();
            loop {
                if let Some(job) = jobs.list.iter_mut().find(|j| j.generation.state == State::Queued) {
                    let g = &mut job.generation;
                    g.state = State::Running;
                    g.steps = Some(IMAGE_STEPS);
                    break (g.id.clone(), g.prompt.clone().unwrap_or_default(), g.seed.unwrap_or(0), g.width.unwrap_or(256), g.height.unwrap_or(256));
                }
                jobs = shared.queued.wait(jobs).unwrap_or_else(|e| e.into_inner());
            }
        };
        let started = Instant::now();
        let timed = |d: Duration| {
            let t = Instant::now();
            std::thread::sleep(d);
            t.elapsed().as_secs_f64()
        };
        let text = timed(shared.config.text);
        let mut steps = Vec::new();
        for k in 1..=IMAGE_STEPS {
            steps.push(timed(shared.config.step));
            shared.update(&id, |j| j.generation.step = Some(k));
        }
        if prompt.contains("[fail]") {
            shared.update(&id, |j| finish(&mut j.generation, State::Failed, started, None, Some("The mock server was asked to fail.".into())));
            continue;
        }
        let t = Instant::now();
        let png = paint(width, height, seed);
        std::thread::sleep(shared.config.decode.saturating_sub(t.elapsed()));
        let decode = t.elapsed().as_secs_f64();
        shared.update(&id, |j| {
            j.png = Some(Arc::new(png));
            finish(&mut j.generation, State::Done, started, Some(Timings { text, steps, decode }), None);
        });
    }
}

fn finish(g: &mut Generation, state: State, started: Instant, timings: Option<Timings>, error: Option<String>) {
    g.state = state;
    g.step = None;
    g.steps = None;
    g.seconds = Some((started.elapsed().as_secs_f64() * 100.0).round() / 100.0);
    g.timings = timings;
    g.error = error;
}

/// A seed-coloured diagonal gradient with rings, so each generation looks different.
pub fn paint(width: u32, height: u32, seed: u64) -> Vec<u8> {
    let hue = |n: u64| (n % 360) as f32;
    let (h1, h2) = (hue(seed.wrapping_mul(2_654_435_761)), hue(seed.wrapping_mul(40_503).wrapping_add(137)));
    let (a, b) = (hsv(h1, 0.55, 0.95), hsv(h2, 0.75, 0.35));
    let (w, h) = (width as f32, height as f32);
    let (cx, cy) = (w * (0.3 + (seed % 5) as f32 * 0.1), h * (0.35 + (seed % 3) as f32 * 0.1));
    let ring = w.min(h) / 9.0;
    let mut rgb = vec![0u8; width as usize * height as usize * 3];
    for (y, line) in rgb.chunks_exact_mut(width as usize * 3).enumerate() {
        let fy = y as f32;
        for (x, px) in line.as_chunks_mut::<3>().0.iter_mut().enumerate() {
            let fx = x as f32;
            let t = (fx / w + fy / h) * 0.5;
            let d = ((fx - cx).powi(2) + (fy - cy).powi(2)).sqrt() / ring;
            let glow = (1.0 - (d.fract() - 0.5).abs() * 2.0).powi(6) * (1.0 - (d / 6.0).min(1.0)) * 0.35;
            for c in 0..3 {
                let v = a[c] * (1.0 - t) + b[c] * t + glow;
                px[c] = (v.clamp(0.0, 1.0) * 255.0) as u8;
            }
        }
    }
    let mut png = Vec::new();
    let encoder = image::codecs::png::PngEncoder::new_with_quality(&mut png, image::codecs::png::CompressionType::Fast, image::codecs::png::FilterType::Sub);
    image::ImageEncoder::write_image(encoder, &rgb, width, height, image::ExtendedColorType::Rgb8).map(|()| png).unwrap_or_default()
}

fn hsv(h: f32, s: f32, v: f32) -> [f32; 3] {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let (r, g, b) = match (h / 60.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    [r + v - c, g + v - c, b + v - c]
}

struct Request {
    method: String,
    path: String,
    content_type: String,
    body: Vec<u8>,
}

fn read_request(reader: &mut BufReader<TcpStream>) -> std::io::Result<Request> {
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_owned();
    let path = parts.next().unwrap_or_default().split('?').next().unwrap_or_default().to_owned();
    let (mut length, mut content_type) = (0usize, String::new());
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header)? == 0 || header.trim().is_empty() {
            break;
        }
        if let Some((name, value)) = header.split_once(':') {
            match name.trim().to_ascii_lowercase().as_str() {
                "content-length" => length = value.trim().parse().unwrap_or(0),
                "content-type" => content_type = value.trim().to_owned(),
                _ => {}
            }
        }
    }
    let mut body = vec![0; length.min(MAX_BODY)];
    reader.read_exact(&mut body)?;
    Ok(Request { method, path, content_type, body })
}

fn respond(stream: &mut TcpStream, status: u16, content_type: &str, body: &[u8]) -> std::io::Result<()> {
    let reason = match status {
        200 => "OK",
        202 => "Accepted",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        409 => "Conflict",
        _ => "Error",
    };
    write!(stream, "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len())?;
    stream.write_all(body)?;
    stream.flush()
}

fn json(stream: &mut TcpStream, status: u16, value: &impl serde::Serialize) -> std::io::Result<()> {
    respond(stream, status, "application/json", &serde_json::to_vec(value).unwrap_or_default())
}

fn text(stream: &mut TcpStream, status: u16, message: &str) -> std::io::Result<()> {
    respond(stream, status, "text/plain; charset=utf-8", message.as_bytes())
}

fn handle(stream: TcpStream, shared: &Shared) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let request = read_request(&mut reader)?;
    let mut stream = stream;
    let segments: Vec<&str> = request.path.trim_matches('/').split('/').collect();
    match (request.method.as_str(), segments.as_slice()) {
        ("GET", ["v1", "status"]) => json(&mut stream, 200, &status(shared)),
        ("GET", ["v1", "generations"]) => {
            let list: Vec<Generation> = shared.lock().list.iter().rev().map(|j| j.generation.clone()).collect();
            json(&mut stream, 200, &list)
        }
        ("POST", ["v1", "generations"]) => match create(shared, &request) {
            Ok(g) => json(&mut stream, 202, &g),
            Err(why) => text(&mut stream, 400, &why),
        },
        ("GET", ["v1", "generations", id]) => match find(shared, id) {
            Some(g) => json(&mut stream, 200, &g),
            None => text(&mut stream, 404, "no such generation"),
        },
        ("DELETE", ["v1", "generations", id]) => {
            let mut jobs = shared.lock();
            match jobs.list.iter().position(|j| j.generation.id == *id) {
                None => text(&mut stream, 404, "no such generation"),
                Some(i) if jobs.list[i].generation.state == State::Running => text(&mut stream, 409, "the generation is running"),
                Some(i) => {
                    jobs.list.remove(i);
                    drop(jobs);
                    respond(&mut stream, 204, "text/plain", b"")
                }
            }
        }
        ("GET", ["v1", "generations", id, "image"]) => {
            let found = shared.lock().list.iter().find(|j| j.generation.id == *id).map(|j| j.png.clone());
            match found {
                None => text(&mut stream, 404, "no such generation"),
                Some(Some(png)) => respond(&mut stream, 200, "image/png", &png),
                Some(None) => text(&mut stream, 409, "the generation is not done"),
            }
        }
        ("GET", ["v1", "generations", _, "video"]) => text(&mut stream, 404, "this server makes images"),
        ("GET", ["v1", "generations", id, "events"]) => events(stream, shared, id),
        _ => text(&mut stream, 404, "no such route"),
    }
}

fn find(shared: &Shared, id: &str) -> Option<Generation> {
    shared.lock().list.iter().find(|j| j.generation.id == id).map(|j| j.generation.clone())
}

fn status(shared: &Shared) -> Status {
    let jobs = shared.lock();
    let running = jobs.list.iter().find(|j| j.generation.state == State::Running);
    let busy = running.is_some();
    Status {
        generating: running.map(|j| j.generation.id.clone()),
        generations: jobs.list.len() as u32,
        memory: Some(Memory { free: TOTAL_MEMORY - (38 << 30) - if busy { 21 << 30 } else { 0 }, total: TOTAL_MEMORY }),
        models: (!busy).then(|| MODELS.iter().map(|m| (*m).to_owned()).collect()),
        queued: jobs.list.iter().filter(|j| j.generation.state == State::Queued).count() as u32,
    }
}

fn create(shared: &Shared, request: &Request) -> Result<Generation, String> {
    let boundary = request.content_type.split("boundary=").nth(1).ok_or("expected multipart/form-data")?.trim_matches('"');
    let fields = parse_multipart(&request.body, boundary);
    if let Some(unknown) = fields.keys().find(|k| !FIELDS.contains(&k.as_str())) {
        return Err(format!("unknown field: {unknown}"));
    }
    let prompt = fields.get("prompt").filter(|p| !p.trim().is_empty()).ok_or("prompt is required")?.clone();
    let seed = fields.get("seed").map_or(Ok(0), |s| s.parse::<u64>().map_err(|_| "seed must be a whole number".to_owned()))?;
    let side = |name: &str| -> Result<u32, String> {
        let raw = fields.get(name).map_or("1024", String::as_str);
        sizes::side(raw).map_err(|e| format!("{name}: {e}"))
    };
    let (width, height) = (side("width")?, side("height")?);
    let mut jobs = shared.lock();
    jobs.next += 1;
    let generation = Generation {
        id: format!("{:08x}{:04x}", (now_secs() as u32).wrapping_mul(2_654_435_761), jobs.next),
        state: State::Queued,
        queued: now_secs(),
        seconds: None,
        step: None,
        steps: None,
        error: None,
        kind: Kind::Image,
        width: Some(width),
        height: Some(height),
        prompt: Some(prompt),
        seed: Some(seed),
        timings: None,
    };
    jobs.list.push(Job { generation: generation.clone(), png: None });
    shared.queued.notify_all();
    Ok(generation)
}

pub fn parse_multipart(body: &[u8], boundary: &str) -> HashMap<String, String> {
    let body = String::from_utf8_lossy(body);
    let delimiter = format!("--{boundary}");
    let mut fields = HashMap::new();
    for part in body.split(delimiter.as_str()).skip(1) {
        if part.starts_with("--") {
            break;
        }
        let part = part.strip_prefix("\r\n").unwrap_or(part);
        let Some((headers, value)) = part.split_once("\r\n\r\n") else { continue };
        let Some(name) = headers.split("name=\"").nth(1).and_then(|s| s.split('"').next()) else { continue };
        fields.insert(name.to_owned(), value.strip_suffix("\r\n").unwrap_or(value).to_owned());
    }
    fields
}

fn events(mut stream: TcpStream, shared: &Shared, id: &str) -> std::io::Result<()> {
    if find(shared, id).is_none() {
        return text(&mut stream, 404, "no such generation");
    }
    write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")?;
    stream.flush()?;
    let mut last = None;
    loop {
        let Some(g) = find(shared, id) else { return Ok(()) };
        let mark = (g.state, g.step);
        let worth = g.is_finished() || (g.state == State::Running && g.step.is_some());
        if worth && last != Some(mark) {
            last = Some(mark);
            write!(stream, "data: {}\n\n", serde_json::to_string(&g).unwrap_or_default())?;
            stream.flush()?;
        }
        if g.is_finished() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(25));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multipart_from_the_client_round_trips() {
        let (content_type, body) = crate::api::multipart(&[("prompt", "a\r\nb".into()), ("seed", "9".into())], "XyZ");
        assert!(content_type.ends_with("boundary=XyZ"));
        let fields = parse_multipart(&body, "XyZ");
        assert_eq!(fields.get("prompt").map(String::as_str), Some("a\r\nb"));
        assert_eq!(fields.get("seed").map(String::as_str), Some("9"));
    }

    #[test]
    fn paint_makes_a_png_of_the_size_asked() {
        let png = paint(320, 192, 7);
        let img = image::load_from_memory_with_format(&png, image::ImageFormat::Png).unwrap();
        assert_eq!((img.width(), img.height()), (320, 192));
    }
}

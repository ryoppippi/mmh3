//! `mmh3 server`, which takes a generation over HTTP and hands back the video or the picture it
//! wrote.
//!
//! A form field stands for the option of the same name, so what a request asks for is turned into
//! the arguments the command would have had and handed to the same generation. There is one
//! device, so there is one generation at a time: requests queue, and the answer to a request is
//! the job it made rather than the video, which is not there yet.
//!
//! There is no authentication here. That is somebody else's to add in front, which is why this
//! listens on loopback unless it is told otherwise.

use crate::cli::{parse_options, split_ffmpeg_arguments};
use axum::extract::{Multipart, Path, State};
use axum::http::{StatusCode, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::stream::{Stream, unfold};
use serde_json::json;
use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::io::AsyncWriteExt;

/// The directories of a job that hold a picture's previews and the references a request uploaded.
const PREVIEWS: &str = "previews";
const REFERENCES: &str = "references";
/// The file of a job that says what it was asked, how it went and what made it, from which a server
/// started again restores the generations of the one before.
const METADATA: &str = "metadata.json";
/// Where a server waits when it is not told.
const DEFAULT_LISTEN: &str = "127.0.0.1:8833";
/// The most a field that stands for an option may hold. An option is a number or a sentence, and
/// a prompt is the longest of them by far.
const FIELD_BYTES: u64 = 1 << 20;
/// The most an uploaded file may hold. A reference clip is the largest thing a request carries.
const FILE_BYTES: u64 = 4 << 30;
/// How often a stream of events looks to see whether the generation it watches has moved. A step
/// is seconds long, so this is often enough to see every one of them and seldom enough to cost
/// nothing between them.
const WATCH: Duration = Duration::from_millis(250);

const USAGE: &str = "usage: mmh3 server [--listen ADDR] [--jobs DIR] [--format mp4|webm] \
                     [--idle-unload SECONDS] [--image-dit FILE] [--image-vae FILE] \
                     [--models DIR] [--dit FILE] [--text-encoder FILE] \
                     [--video-vae FILE] [--audio-vae FILE] [--devices CARD,CARD...] \
                     [--worker HOST[:PORT] [--worker-units UNITS]]... [--local-worker [--worker-units UNITS]] \
                     [--token FILE] [--vram-budget GB] [--consistent] [--ffmpeg ARGS...]";
/// The options that are the server's own rather than a generation's. `--image-dit` and
/// `--image-vae` are a picture's `--dit` and `--vae`.
const SERVER_OPTIONS: &[&str] = &[
    "listen",
    "jobs",
    "format",
    "idle-unload",
    "image-dit",
    "image-vae",
];
/// The machine options a picture is told too, which are the ones `mmh3 image` takes.
const IMAGE_MACHINE_OPTIONS: &[&str] = &["models", "text-encoder", "vram-budget"];
/// What a request for a picture may give: the options of `mmh3 image` about what to generate.
const IMAGE_OPTIONS: &[&str] = &[
    "prompt",
    "prompt-file",
    "reference",
    "width",
    "height",
    "min-generation-area",
    "seed",
    "attention-precision",
    "attention",
    "sparse-tau",
    "sparse-start",
    "draft-steps",
    "latent-upscaler",
    "renoise-sigma",
    "steps",
    "shift",
    "patch",
    "lora",
    "lora-strength",
    "lora-mode",
];
/// The options of a picture whose value is a file, which a request uploads.
const IMAGE_FILE_OPTIONS: &[&str] = &["prompt-file", "reference"];
/// The options of a generation that say what this machine has and which machines compute, which
/// the server is given and a request may not give. `--local-worker` and `--consistent`, which a
/// form cannot give either, are the others.
const MACHINE_OPTIONS: &[&str] = &[
    "models",
    "dit",
    "text-encoder",
    "video-vae",
    "audio-vae",
    "devices",
    "vram-budget",
    "worker",
    "worker-units",
    "token",
];
/// The options of a generation whose value is a file, which a request uploads.
const FILE_OPTIONS: &[&str] = &[
    "prompt-file",
    "context",
    "first-frame",
    "last-frame",
    "reference",
    "reference-audio",
    "reference-video",
];
/// The options of a generation whose value names a file in the server's models directory, with
/// the directories it is looked for in, in order.
const MODEL_FILE_OPTIONS: &[(&str, &[&str])] = &[
    ("patch", &["patches", "loras"]),
    ("lora", &["loras"]),
    ("latent-upscaler", &["latent_upscale_models"]),
];

/// The container every generation of a server is written in.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Container {
    Mp4,
    Webm,
}

impl Container {
    fn extension(self) -> &'static str {
        match self {
            Container::Mp4 => "mp4",
            Container::Webm => "webm",
        }
    }

    fn content_type(self) -> &'static str {
        match self {
            Container::Mp4 => "video/mp4",
            Container::Webm => "video/webm",
        }
    }
}

/// What a generation writes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Output {
    Video,
    Picture,
}

impl Output {
    fn name(self) -> &'static str {
        match self {
            Output::Video => "video",
            Output::Picture => "image",
        }
    }
}

/// What a generation the server was asked for is doing.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    Queued,
    Running,
    Done,
    Failed,
    Cancelled,
}

impl Stage {
    fn name(self) -> &'static str {
        match self {
            Stage::Queued => "queued",
            Stage::Running => "running",
            Stage::Done => "done",
            Stage::Failed => "failed",
            Stage::Cancelled => "cancelled",
        }
    }

    fn ended(self) -> bool {
        matches!(self, Stage::Done | Stage::Failed | Stage::Cancelled)
    }
}

/// One generation the server was asked for.
struct Job {
    state: Stage,
    /// What the generation was told, in the form the command would have been told it.
    arguments: Vec<String>,
    /// Where this job's uploads and what it wrote live.
    directory: PathBuf,
    output: Output,
    /// Where this job's video or picture was written.
    written: PathBuf,
    /// Why it failed, for a client that asks after it did.
    failure: Option<String>,
    /// What the request asked for, in the terms a client shows it in.
    request: serde_json::Value,
    queued: SystemTime,
    started: Option<Instant>,
    started_at: Option<SystemTime>,
    finished_at: Option<SystemTime>,
    seconds: Option<f64>,
    /// The seconds of each stage, as a picture's run reports them.
    timings: serde_json::Value,
    /// The last step with a preview in the job's `previews` directory.
    preview_step: Option<usize>,
    /// The checkpoints the run read.
    models: Vec<PathBuf>,
}

impl Job {
    fn new(output: Output, arguments: Vec<String>, directory: PathBuf, written: PathBuf) -> Self {
        Job {
            state: Stage::Queued,
            request: request(&arguments, output),
            arguments,
            directory,
            output,
            written,
            failure: None,
            queued: SystemTime::now(),
            started: None,
            started_at: None,
            finished_at: None,
            seconds: None,
            timings: serde_json::Value::Null,
            preview_step: None,
            models: Vec::new(),
        }
    }

    fn describe(&self, id: &str) -> serde_json::Value {
        // Only the generation that is running is the one the steps belong to.
        let progress = (self.state == Stage::Running)
            .then(crate::generation::progress)
            .flatten();
        json!({
            "id": id,
            "kind": self.output.name(),
            "state": self.state.name(),
            "queued": self.queued.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
            "seconds": self.seconds,
            "step": progress.map(|progress| progress.step),
            "steps": progress.map(|progress| progress.steps),
            "error": self.failure,
            "request": self.request,
            "created_at": unix_seconds(self.queued),
            "started_at": self.started_at.map(unix_seconds),
            "finished_at": self.finished_at.map(unix_seconds),
            "timings": self.timings,
            "preview_step": self.preview_step,
        })
    }

    /// Takes in what a picture's run reports of itself.
    fn take_report(&mut self, report: &crate::image::Report) {
        let timings = &report.timings;
        self.timings = json!({
            "load_text": timings.load_text,
            "encode": timings.encode,
            "load_dit": timings.load_dit,
            "steps": timings.steps,
            "decode_read": timings.decode_read,
            "decode": timings.decode,
            "write": timings.write,
        });
        self.preview_step = report.preview_step;
        self.models = report.models.clone();
        if let Some(precision) = report.attention_precision {
            self.request["attention_precision"] = precision.into();
        }
    }
}

/// A job of an earlier run of the server from what its directory's metadata says, or None for a
/// directory that is not a job's. A job that was waiting or running when that server stopped never
/// ended, and is restored failed.
fn restore(directory: &std::path::Path) -> Option<Job> {
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join(METADATA)).ok()?).ok()?;
    let output = match metadata["kind"].as_str()? {
        "image" => Output::Picture,
        "video" => Output::Video,
        _ => return None,
    };
    let (state, failure) = match metadata["state"].as_str()? {
        "done" => (Stage::Done, None),
        "failed" => (Stage::Failed, metadata["error"].as_str().map(str::to_owned)),
        "cancelled" => (Stage::Cancelled, None),
        "queued" | "running" => (Stage::Failed, Some(INTERRUPTED.to_owned())),
        _ => return None,
    };
    let written = match output {
        Output::Picture => directory.join("image.png"),
        // The container is the one of the server that wrote it, which need not be this one's.
        Output::Video => std::fs::read_dir(directory)
            .ok()?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .find(|path| path.file_stem().is_some_and(|stem| stem == "video"))
            .unwrap_or_else(|| directory.join("video")),
    };
    let time = |field: &str| {
        metadata[field]
            .as_f64()
            .filter(|seconds| seconds.is_finite() && *seconds >= 0.0)
            .map(|seconds| UNIX_EPOCH + Duration::from_secs_f64(seconds))
    };
    Some(Job {
        state,
        arguments: Vec::new(),
        directory: directory.to_owned(),
        output,
        written,
        failure,
        request: metadata["request"].clone(),
        queued: time("created_at")?,
        started: None,
        started_at: time("started_at"),
        finished_at: time("finished_at"),
        seconds: metadata["seconds"].as_f64(),
        timings: metadata["timings"].clone(),
        preview_step: metadata["preview_step"].as_u64().map(|step| step as usize),
        models: Vec::new(),
    })
}

/// Every job under the jobs directory `directory` that a server before this one left, by id.
fn restore_all(directory: &std::path::Path) -> HashMap<String, Job> {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return HashMap::new();
    };
    entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let id = entry.file_name().into_string().ok()?;
            let job = restore(&entry.path())?;
            // What the metadata said of a job that never ended is put right, so that it says so
            // the next time too.
            if job.failure.as_deref() == Some(INTERRUPTED) {
                write_metadata(&job.directory, job.describe(&id), &[]);
            }
            Some((id, job))
        })
        .collect()
}

/// Why a job restored waiting or running failed.
const INTERRUPTED: &str = "the server stopped before this generation ended";

/// Writes a job's `metadata.json` beside what it wrote: `described`, the generation as the server
/// describes it, with the checkpoints it read, the build of mmh3 and the machine. It is written
/// beside and renamed, so that a server starting never reads half of one. A job that cannot have
/// its metadata written still runs, and says so.
fn write_metadata(
    directory: &std::path::Path,
    mut described: serde_json::Value,
    models: &[PathBuf],
) {
    described["models"] = models
        .iter()
        .map(|path| crate::provenance::model(path))
        .collect();
    described["mmh3"] = crate::provenance::build();
    described["machine"] = crate::provenance::machine();
    let partial = directory.join(format!(".{METADATA}"));
    let written = serde_json::to_vec_pretty(&described)
        .map_err(std::io::Error::other)
        .and_then(|bytes| std::fs::write(&partial, bytes))
        .and_then(|()| std::fs::rename(&partial, directory.join(METADATA)));
    if let Err(error) = written {
        eprintln!(
            "warning: writing the metadata of {}: {error}",
            directory.display()
        );
    }
}

fn unix_seconds(time: SystemTime) -> f64 {
    time.duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

/// What a generation told `arguments` was asked for, as `{prompt, width, height, seed,
/// min_generation_area, attention_precision, attention, sparse_tau, sparse_start, draft_steps,
/// latent_upscaler, renoise_sigma, steps, shift, patch, lora, lora_strength, lora_mode,
/// references}`, with a picture's defaults where the request left an option out. A prompt from a
/// file is the file's text, a reference is the name of its file in the job's `references`
/// directory, and a latent upscaler, a patch or a LoRA is the name of its checkpoint.
fn request(arguments: &[String], output: Output) -> serde_json::Value {
    let given = |name: &str| {
        let option = format!("--{name}");
        arguments
            .windows(2)
            .rev()
            .find(|pair| pair[0] == option)
            .map(|pair| pair[1].as_str())
    };
    let number = |name: &str, default: Option<usize>| -> serde_json::Value {
        match given(name) {
            Some(value) => value
                .parse::<u64>()
                .map_or_else(|_| value.into(), serde_json::Value::from),
            None => default.into(),
        }
    };
    let float = |name: &str| -> serde_json::Value {
        given(name).map_or(serde_json::Value::Null, |value| {
            value
                .parse::<f64>()
                .map_or_else(|_| value.into(), serde_json::Value::from)
        })
    };
    let file_name = |path: &str| {
        std::path::Path::new(path)
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
    };
    let picture = output == Output::Picture;
    let prompt = given("prompt")
        .map(str::to_owned)
        .or_else(|| given("prompt-file").and_then(|path| std::fs::read_to_string(path).ok()));
    let references: Vec<String> = crate::cli::option_values(arguments, "reference")
        .into_iter()
        .filter_map(file_name)
        .collect();
    let lora = given("lora").and_then(file_name);
    json!({
        "prompt": prompt,
        "width": number("width", picture.then_some(crate::image::DEFAULT_SIDE)),
        "height": number("height", picture.then_some(crate::image::DEFAULT_SIDE)),
        "seed": number("seed", Some(0)),
        "min_generation_area": number(
            "min-generation-area",
            picture.then_some(crate::image::DEFAULT_MIN_GENERATION_AREA),
        ),
        "attention_precision": given("attention-precision"),
        // A video's default depends on its DiT, which picks VSA when it has VSA's gates.
        "attention": given("attention").or(picture.then_some("dense")),
        "sparse_tau": float("sparse-tau"),
        "sparse_start": float("sparse-start"),
        "draft_steps": number("draft-steps", picture.then_some(0)),
        "latent_upscaler": given("latent-upscaler").and_then(file_name),
        "renoise_sigma": float("renoise-sigma"),
        "steps": number("steps", None),
        "shift": float("shift"),
        "patch": given("patch").and_then(file_name),
        "lora": lora,
        "lora_strength": match (&lora, given("lora-strength")) {
            (None, _) => serde_json::Value::Null,
            (Some(_), None) => 1.0.into(),
            (Some(_), Some(_)) => float("lora-strength"),
        },
        "lora_mode": lora.as_ref().map(|_| given("lora-mode").unwrap_or("adapter")),
        "references": references,
    })
}

/// Everything the handlers share.
struct Server {
    jobs: Mutex<HashMap<String, Job>>,
    /// Ids in the order they were asked for, which is the order they run in.
    queue: Sender<String>,
    /// What every generation is told before what its own request said: the machine options the
    /// server was given.
    defaults: Vec<String>,
    /// The same for a picture: the machine options `mmh3 image` takes, with `--image-dit` and
    /// `--image-vae` as its `--dit` and `--vae`.
    image_defaults: Vec<String>,
    /// Where a request's `lora` and `patch` are looked for.
    models: Option<PathBuf>,
    /// What every generation hands ffmpeg to write its video, after everything else it is told, or
    /// none for the native output.
    ffmpeg: Option<Vec<String>>,
    container: Container,
    directory: PathBuf,
    /// What the stream of every generation says next, as an event's name and its data.
    announcements: tokio::sync::broadcast::Sender<Announcement>,
}

/// An event of `GET /v1/events`: its name and its data.
type Announcement = (&'static str, serde_json::Value);

/// How many events a client of `GET /v1/events` may fall behind by before it is sent a snapshot in
/// place of the ones it missed.
const ANNOUNCEMENTS: usize = 256;
/// How often `GET /v1/events` says something to a client when nothing happens, so that a proxy
/// between them does not take the stream for a dead one.
const KEEP_ALIVE: Duration = Duration::from_secs(15);

/// `mmh3 server [--listen ADDR] [--jobs DIR] ...`, which serves until it is stopped.
pub fn serve(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let (arguments, ffmpeg) = split_ffmpeg_arguments(arguments);
    if arguments.iter().any(|argument| argument == "--out") {
        return Err("the server names every job's video itself, so leave out --out".into());
    }
    if let Some(name) = arguments
        .iter()
        .filter_map(|argument| argument.strip_prefix("--"))
        .find(|name| crate::generation::OPTIONS.contains(name) && !MACHINE_OPTIONS.contains(name))
    {
        return Err(format!("--{name} is for each request to give, not the server").into());
    }
    let options = parse_options(
        arguments,
        &[SERVER_OPTIONS, MACHINE_OPTIONS].concat(),
        crate::generation::FLAGS,
        USAGE,
    )?;
    let listen = options
        .get("listen")
        .map_or(DEFAULT_LISTEN, |listen| *listen)
        .to_owned();
    let directory = match options.get("jobs") {
        Some(directory) => PathBuf::from(directory),
        None => {
            crate::models::cache_file("jobs").ok_or("pass a directory for the jobs with --jobs")?
        }
    };
    std::fs::create_dir_all(&directory)?;
    let restored = restore_all(&directory);
    let container = match options.get("format").copied().unwrap_or("mp4") {
        "mp4" => Container::Mp4,
        "webm" => Container::Webm,
        other => return Err(format!("--format must be mp4 or webm, not {other}").into()),
    };
    // Checked here rather than by the first generation, which would fail only after its steps.
    if container == Container::Webm {
        match ffmpeg {
            None if !cfg!(feature = "webm") => {
                return Err("--format webm needs a build with webm, or --ffmpeg".into());
            }
            // ffmpeg's recipe without arguments is H.264 and AAC, which WebM cannot hold.
            Some([]) => {
                return Err(
                    "--format webm with --ffmpeg needs ffmpeg's arguments for WebM, \
                            such as -c:v libvpx-vp9 -c:a libopus"
                        .into(),
                );
            }
            _ => {}
        }
    }

    // Everything but the server's own options is about this machine, and every generation is
    // told it. A picture is told what `mmh3 image` takes of it.
    let mut defaults = Vec::new();
    let mut image_defaults = Vec::new();
    let mut remaining = arguments.iter();
    while let Some(name) = remaining.next() {
        // An option that stands alone carries nothing after it to pair with.
        if crate::generation::FLAGS.contains(&&name[2..]) {
            defaults.push(name.clone());
            if name == "--consistent" {
                image_defaults.push(name.clone());
            }
            continue;
        }
        let Some(value) = remaining.next() else { break };
        let option = &name[2..];
        match option {
            "image-dit" | "image-vae" => {
                image_defaults.push(name.replacen("image-", "", 1));
                image_defaults.push(value.clone());
            }
            _ if SERVER_OPTIONS.contains(&option) => {}
            _ => {
                defaults.push(name.clone());
                defaults.push(value.clone());
                if IMAGE_MACHINE_OPTIONS.contains(&option) {
                    image_defaults.push(name.clone());
                    image_defaults.push(value.clone());
                }
            }
        }
    }
    crate::generation::Settings::check_machines(&options, &defaults)?;
    let idle_unload = crate::cli::option_number(&options, "idle-unload", 0)?;
    #[cfg(any(feature = "cuda", feature = "metal"))]
    if idle_unload > 0 {
        crate::resident::release_when_idle(Duration::from_secs(idle_unload as u64));
    }
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    if idle_unload > 0 {
        return Err(
            "this build holds no models, so it has none for --idle-unload to let go of".into(),
        );
    }
    // NOTE: a run that has nothing to hand its work to reads every model itself and lets go of
    // them when it ends, which is right for a command and wrong for a server. A worker is what
    // holds models between the runs that use them, with the budget it is held to and the letting
    // go it does when a card has no memory left, so a run here that names no worker lends this
    // machine's cards through the one worker in this process, even a run with one card. A server
    // told to use machines of its own is left alone: it lends its cards only with --local-worker.
    crate::generation::keep_models();

    let (queue, waiting) = channel();
    let server = Arc::new(Server {
        jobs: Mutex::new(restored),
        queue,
        defaults,
        image_defaults,
        models: crate::models::models_directory(&options),
        ffmpeg: ffmpeg.map(<[String]>::to_vec),
        container,
        directory: directory.clone(),
        announcements: tokio::sync::broadcast::channel(ANNOUNCEMENTS).0,
    });

    // The generations run here, one after another, off the runtime that answers the requests. A
    // generation holds the device for minutes and answers nothing while it does.
    let generating = Arc::clone(&server);
    std::thread::Builder::new()
        .name("generate".to_owned())
        .spawn(move || {
            for id in waiting {
                generate(&generating, &id, |arguments, output, watch| match output {
                    Output::Video => crate::generate::run(arguments, USAGE),
                    Output::Picture => crate::image::run_watched(arguments, watch),
                });
            }
        })?;

    let application = Router::new()
        .route("/v1/generations", post(create).get(list))
        .route("/v1/images", post(create_image))
        .route("/v1/status", get(status))
        .route("/v1/generations/{id}", get(describe).delete(forget))
        .route("/v1/generations/{id}/cancel", post(cancel))
        .route("/v1/generations/{id}/events", get(events))
        .route("/v1/generations/{id}/video", get(video))
        .route("/v1/generations/{id}/image", get(image))
        .route("/v1/generations/{id}/preview", get(latest_preview))
        .route("/v1/generations/{id}/preview/{step}", get(preview))
        .route("/v1/events", get(all_events))
        // Each field holds itself to FIELD_BYTES or FILE_BYTES as it is read, which axum's limit on
        // the whole body, 2 MB, would cut short of any picture worth editing.
        .layer(axum::extract::DefaultBodyLimit::disable())
        .with_state(Arc::clone(&server));

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let listener = tokio::net::TcpListener::bind(&listen).await?;
        println!("serving on {listen}, jobs in {}", directory.display());
        tokio::spawn(announce(server));
        axum::serve(listener, application).await
    })?;
    Ok(())
}

/// Where a picture's run reports what it has done so far.
type Reporting<'a> = dyn FnMut(&crate::image::Report) + 'a;

/// Runs one job to its end, whichever end that is, with `run`, which is handed what the generation
/// was told, what it writes and where a picture reports itself.
fn generate(
    server: &Server,
    id: &str,
    run: impl FnOnce(&[String], Output, &mut Reporting) -> Result<(), Box<dyn Error>>,
) {
    let (arguments, output, directory, described) = {
        let mut jobs = server.jobs.lock().expect("the jobs");
        let Some(job) = jobs.get_mut(id) else {
            return;
        };
        // A job cancelled while it waited is dropped here, with its metadata written as for any
        // other end.
        if job.state != Stage::Queued {
            let (directory, described) = (job.directory.clone(), job.describe(id));
            drop(jobs);
            write_metadata(&directory, described, &[]);
            return;
        }
        job.state = Stage::Running;
        job.started = Some(Instant::now());
        job.started_at = Some(SystemTime::now());
        // Under the lock, so that a cancel of this job, which takes the lock to see it running,
        // comes after the flag of the run before it is cleared.
        crate::generation::starting();
        (
            job.arguments.clone(),
            job.output,
            job.directory.clone(),
            job.describe(id),
        )
    };
    println!("generation {id} begins");
    // Written as it starts too, so that a server stopped during it restores it as failed.
    write_metadata(&directory, described, &[]);
    let mut watch = |report: &crate::image::Report| {
        if let Some(job) = server.jobs.lock().expect("the jobs").get_mut(id) {
            job.take_report(report);
        }
    };
    let generated = run(&arguments, output, &mut watch);
    let mut jobs = server.jobs.lock().expect("the jobs");
    let Some(job) = jobs.get_mut(id) else {
        return;
    };
    job.seconds = job.started.map(|started| started.elapsed().as_secs_f64());
    job.finished_at = Some(SystemTime::now());
    match generated {
        Ok(()) => {
            job.state = Stage::Done;
            println!("generation {id} wrote {}", job.written.display());
        }
        Err(error) if error.is::<crate::generation::Cancelled>() => {
            job.state = Stage::Cancelled;
            println!("generation {id} was cancelled");
        }
        Err(error) => {
            job.state = Stage::Failed;
            println!("generation {id} failed: {error}");
            job.failure = Some(error.to_string());
        }
    }
    let (described, models) = (job.describe(id), job.models.clone());
    drop(jobs);
    write_metadata(&directory, described, &models);
}

/// Takes a generation, and answers with the job rather than with the video, which is not there
/// yet. The form's fields are the options of `mmh3 generate` but the machine's. A file is written
/// beside the job and stands for the path it was written to, and a `lora` or `patch` names a file
/// in the server's models directory.
async fn create(
    State(server): State<Arc<Server>>,
    mut form: Multipart,
) -> Result<Response, Failure> {
    let id = next_id();
    let directory = server.directory.join(&id);
    tokio::fs::create_dir_all(&directory).await?;
    let video = directory.join(format!("video.{}", server.container.extension()));
    let mut arguments = server.defaults.clone();
    arguments.push("--out".to_owned());
    arguments.push(video.to_string_lossy().into_owned());

    if let Err(failure) =
        read_fields(&server, &mut form, &directory, &mut arguments, field_kind).await
    {
        // A refused request leaves nothing behind, not even the files it sent before the field
        // that was refused.
        let _ = tokio::fs::remove_dir_all(&directory).await;
        return Err(failure);
    }
    // Last, since generate takes everything after --ffmpeg as ffmpeg's.
    if let Some(ffmpeg) = &server.ffmpeg {
        arguments.push("--ffmpeg".to_owned());
        arguments.extend(ffmpeg.iter().cloned());
    }

    queue_job(&server, id, Output::Video, arguments, directory, video)
}

/// Takes a picture with MiniMax H3 Image, the way `create` takes a video: the form's fields are
/// the options of `mmh3 image` about what to generate, and the answer is the job. The picture is
/// a PNG.
async fn create_image(
    State(server): State<Arc<Server>>,
    mut form: Multipart,
) -> Result<Response, Failure> {
    let id = next_id();
    let directory = server.directory.join(&id);
    tokio::fs::create_dir_all(&directory).await?;
    let picture = directory.join("image.png");
    let mut arguments = server.image_defaults.clone();
    arguments.push("--out".to_owned());
    arguments.push(picture.to_string_lossy().into_owned());
    arguments.push("--preview-dir".to_owned());
    arguments.push(directory.join(PREVIEWS).to_string_lossy().into_owned());
    if let Err(failure) = read_fields(
        &server,
        &mut form,
        &directory,
        &mut arguments,
        image_field_kind,
    )
    .await
    {
        let _ = tokio::fs::remove_dir_all(&directory).await;
        return Err(failure);
    }
    queue_job(&server, id, Output::Picture, arguments, directory, picture)
}

/// Queues a job and answers with it.
fn queue_job(
    server: &Server,
    id: String,
    output: Output,
    arguments: Vec<String>,
    directory: PathBuf,
    written: PathBuf,
) -> Result<Response, Failure> {
    let job = Job::new(output, arguments, directory, written);
    let described = job.describe(&id);
    server
        .jobs
        .lock()
        .expect("the jobs")
        .insert(id.clone(), job);
    if server.queue.send(id.clone()).is_err() {
        return Err(Failure::gone("nothing is generating any more"));
    }
    Ok((
        StatusCode::ACCEPTED,
        [(header::LOCATION, format!("/v1/generations/{id}"))],
        Json(described),
    )
        .into_response())
}

/// Adds the fields of a request's form to `arguments`, writing its files into `directory`. `kinds`
/// says what each field takes, or why a request may not give it.
async fn read_fields(
    server: &Server,
    form: &mut Multipart,
    directory: &std::path::Path,
    arguments: &mut Vec<String>,
    kinds: fn(&str) -> Result<FieldKind, String>,
) -> Result<(), Failure> {
    while let Some(mut field) = form.next_field().await.map_err(Failure::form)? {
        let Some(name) = field.name().map(str::to_owned) else {
            return Err(Failure::asked("a form field with no name"));
        };
        let kind = kinds(&name).map_err(|message| Failure::asked(&message))?;
        let value = match (kind, field.file_name().map(str::to_owned)) {
            // A file stands for the path it is written to, which is this job's own directory.
            (FieldKind::File, Some(filename)) => {
                let path = upload_path(directory, &name, &named(&filename)?).await?;
                let mut file = tokio::fs::File::create(&path).await?;
                let mut written = 0;
                while let Some(chunk) = field.chunk().await.map_err(Failure::form)? {
                    written += chunk.len() as u64;
                    if written > FILE_BYTES {
                        return Err(Failure::asked("a file longer than this takes"));
                    }
                    file.write_all(&chunk).await?;
                }
                file.flush().await?;
                path.to_string_lossy().into_owned()
            }
            (FieldKind::File, None) => {
                return Err(Failure::asked(&format!("{name} takes an uploaded file")));
            }
            (_, Some(_)) => {
                return Err(Failure::asked(&format!("{name} takes text, not a file")));
            }
            (kind, None) => {
                let mut bytes = Vec::new();
                while let Some(chunk) = field.chunk().await.map_err(Failure::form)? {
                    if (bytes.len() + chunk.len()) as u64 > FIELD_BYTES {
                        return Err(Failure::asked("a field longer than this takes"));
                    }
                    bytes.extend_from_slice(&chunk);
                }
                let text = String::from_utf8(bytes)
                    .map_err(|_| Failure::asked("a form field that is not text"))?;
                match kind {
                    FieldKind::ModelFile(directories) => {
                        model_named(server.models.as_deref(), &name, &text, directories)
                            .map_err(|message| Failure::asked(&message))?
                    }
                    _ => text,
                }
            }
        };
        arguments.push(format!("--{name}"));
        arguments.push(value);
    }
    Ok(())
}

/// Takes a generation away, with whatever it wrote. A generation that is running is left alone,
/// since throwing away what it is about to write while it writes is worse than waiting: cancel it
/// first.
async fn forget(
    State(server): State<Arc<Server>>,
    Path(id): Path<String>,
) -> Result<Response, Failure> {
    let directory = {
        let mut jobs = server.jobs.lock().expect("the jobs");
        match jobs.get(&id).ok_or_else(Failure::unknown)?.state {
            Stage::Running => return Err(Failure::gone("that generation is running")),
            _ => jobs.remove(&id).expect("just looked").directory,
        }
    };
    // A job that never wrote anything has nothing here, which is not a failure to remove it.
    if let Err(error) = tokio::fs::remove_dir_all(&directory).await
        && error.kind() != std::io::ErrorKind::NotFound
    {
        return Err(error.into());
    }
    Ok(StatusCode::NO_CONTENT.into_response())
}

/// Stops a generation: one waiting its turn is dropped from the queue, and one running stops at the
/// boundary of its next step. Either keeps what it wrote, and is `cancelled` once it has stopped.
async fn cancel(
    State(server): State<Arc<Server>>,
    Path(id): Path<String>,
) -> Result<Response, Failure> {
    let mut jobs = server.jobs.lock().expect("the jobs");
    let job = jobs.get_mut(&id).ok_or_else(Failure::unknown)?;
    match job.state {
        Stage::Queued => {
            job.state = Stage::Cancelled;
            job.finished_at = Some(SystemTime::now());
        }
        Stage::Running => crate::generation::cancel(),
        Stage::Cancelled => {}
        Stage::Done | Stage::Failed => return Err(Failure::gone("that generation has ended")),
    }
    Ok((StatusCode::ACCEPTED, Json(job.describe(&id))).into_response())
}

/// Every generation this server has been asked for, the most recent first.
async fn list(State(server): State<Arc<Server>>) -> Response {
    Json(generations(&server)).into_response()
}

/// `{generations: [...]}`, every generation the most recent first.
fn generations(server: &Server) -> serde_json::Value {
    let jobs = server.jobs.lock().expect("the jobs");
    // Ids count up within a run of the server, which orders two asked for in the same instant.
    let mut described: Vec<(SystemTime, &String, serde_json::Value)> = jobs
        .iter()
        .map(|(id, job)| (job.queued, id, job.describe(id)))
        .collect();
    described
        .sort_by(|(one, one_id, _), (other, other_id, _)| (other, other_id).cmp(&(one, one_id)));
    let generations: Vec<serde_json::Value> =
        described.into_iter().map(|(_, _, value)| value).collect();
    json!({ "generations": generations })
}

/// What has changed of the server's generations since `said`, which is what was last said of
/// each, as the events of `GET /v1/events`: a `generation` for each one that changed, and a
/// `preview` for each with a new preview.
fn changes(server: &Server, said: &mut HashMap<String, serde_json::Value>) -> Vec<Announcement> {
    let jobs = server.jobs.lock().expect("the jobs");
    said.retain(|id, _| jobs.contains_key(id));
    let mut changes = Vec::new();
    for (id, job) in jobs.iter() {
        let described = job.describe(id);
        let before = said.get(id);
        if before == Some(&described) {
            continue;
        }
        let preview_step = described["preview_step"].as_u64();
        if preview_step.is_some()
            && before.is_none_or(|before| before["preview_step"].as_u64() != preview_step)
        {
            changes.push((
                "preview",
                json!({
                    "id": id,
                    "step": preview_step,
                    "steps": described["steps"],
                    "url": format!("/v1/generations/{id}/preview/{}", preview_step.unwrap_or(0)),
                }),
            ));
        }
        changes.push(("generation", described.clone()));
        said.insert(id.clone(), described);
    }
    // A generation's own change before the preview it brought, for a client that draws the
    // preview of the step it was told of.
    changes.sort_by_key(|(name, _)| *name == "preview");
    changes
}

/// Tells the clients of `GET /v1/events` what changed, as often as a client watching one
/// generation looks.
async fn announce(server: Arc<Server>) {
    let mut said = HashMap::new();
    // What was there before anybody could listen is in every client's snapshot.
    changes(&server, &mut said);
    loop {
        tokio::time::sleep(WATCH).await;
        for announcement in changes(&server, &mut said) {
            // Nobody listening is not a failure to say it.
            let _ = server.announcements.send(announcement);
        }
    }
}

/// Every generation as it happens, in one stream: a `snapshot` of all of them first, then a
/// `generation` whenever one changes and a `preview` whenever one has a new preview. A client
/// that falls behind is sent another snapshot.
async fn all_events(
    State(server): State<Arc<Server>>,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    use futures_util::StreamExt;

    // Listening before the snapshot is taken, so that nothing falls between the two.
    let listening = server.announcements.subscribe();
    let snapshot = Event::default()
        .event("snapshot")
        .data(generations(&server).to_string());
    let announcements = unfold((server, listening), |(server, mut listening)| async move {
        use tokio::sync::broadcast::error::RecvError;

        let event = match listening.recv().await {
            Ok((name, data)) => Event::default().event(name).data(data.to_string()),
            Err(RecvError::Lagged(_)) => Event::default()
                .event("snapshot")
                .data(generations(&server).to_string()),
            Err(RecvError::Closed) => return None,
        };
        Some((Ok(event), (server, listening)))
    });
    let events = futures_util::stream::once(std::future::ready(Ok(snapshot))).chain(announcements);
    Sse::new(events).keep_alive(KeepAlive::new().interval(KEEP_ALIVE).text("keep-alive"))
}

/// What this machine is doing and what it is holding, for a client deciding whether to ask it for
/// anything.
async fn status(State(server): State<Arc<Server>>) -> Response {
    let (generating, queued, generations) = {
        let jobs = server.jobs.lock().expect("the jobs");
        let generating = jobs
            .iter()
            .find(|(_, job)| job.state == Stage::Running)
            .map(|(id, _)| id.clone());
        let queued = jobs
            .values()
            .filter(|job| job.state == Stage::Queued)
            .count();
        (generating, queued, jobs.len())
    };
    Json(json!({
        "generating": generating,
        "queued": queued,
        "generations": generations,
        // None rather than an empty list: a generation has the models, and saying it holds
        // nothing would be saying something else.
        "models": kept(),
        "memory": memory().map(|(free, total)| json!({ "free": free, "total": total })),
    }))
    .into_response()
}

/// What this machine is holding on to, or None while a generation has the models or on a build
/// that holds none of its own.
fn kept() -> Option<Vec<&'static str>> {
    #[cfg(any(feature = "cuda", feature = "metal"))]
    {
        crate::resident::kept()
    }
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    {
        None
    }
}

/// What the devices of this machine say they have left and what they have in all, between them, or
/// None on a build with neither backend, which has no device to ask.
fn memory() -> Option<(u64, u64)> {
    #[cfg(feature = "cuda")]
    {
        let mut sums = (0u64, 0u64);
        for device in 0..crate::resident::device_count() {
            let (free, total) = mmh3_cuda::memory_info_on(device).ok()?;
            sums = (sums.0 + free as u64, sums.1 + total as u64);
        }
        Some(sums)
    }
    #[cfg(feature = "metal")]
    {
        let (free, total) = mmh3_metal::memory_info().ok()?;
        Some((free as u64, total as u64))
    }
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    {
        None
    }
}

/// What a generation is doing.
async fn describe(
    State(server): State<Arc<Server>>,
    Path(id): Path<String>,
) -> Result<Response, Failure> {
    let jobs = server.jobs.lock().expect("the jobs");
    let job = jobs.get(&id).ok_or_else(Failure::unknown)?;
    Ok(Json(job.describe(&id)).into_response())
}

/// What a client watching one generation is watching.
struct Watch {
    server: Arc<Server>,
    id: String,
    /// The last thing said, so that nothing is said twice.
    said: Option<serde_json::Value>,
    ended: bool,
}

/// A generation as it happens, so that a client watches rather than asks again and again. The
/// first event says where the generation is now, every one after it says what changed, and the
/// stream ends when the generation does.
async fn events(
    State(server): State<Arc<Server>>,
    Path(id): Path<String>,
) -> Result<Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>>, Failure> {
    // A generation nobody has is answered now, where a client can read the answer, rather than
    // inside a stream that would open and say nothing.
    server
        .jobs
        .lock()
        .expect("the jobs")
        .contains_key(&id)
        .then_some(())
        .ok_or_else(Failure::unknown)?;
    let watch = Watch {
        server,
        id,
        said: None,
        ended: false,
    };
    let events = unfold(watch, |mut watch| async move {
        if watch.ended {
            return None;
        }
        loop {
            let described = {
                let jobs = watch.server.jobs.lock().expect("the jobs");
                jobs.get(&watch.id)
                    .map(|job| (job.state, job.describe(&watch.id)))
            };
            // The generation was taken away while somebody watched it, so there is nothing more
            // to say about it.
            let (state, described) = described?;
            if watch.said.as_ref() != Some(&described) {
                watch.ended = state.ended();
                watch.said = Some(described.clone());
                return Some((Ok(Event::default().data(described.to_string())), watch));
            }
            tokio::time::sleep(WATCH).await;
        }
    });
    Ok(Sse::new(events).keep_alive(KeepAlive::default()))
}

/// The video a generation wrote, once it has written one.
async fn video(
    State(server): State<Arc<Server>>,
    Path(id): Path<String>,
) -> Result<Response, Failure> {
    let content_type = server.container.content_type();
    written(&server, &id, Output::Video, content_type).await
}

/// The picture a generation wrote, once it has written one.
async fn image(
    State(server): State<Arc<Server>>,
    Path(id): Path<String>,
) -> Result<Response, Failure> {
    written(&server, &id, Output::Picture, "image/png").await
}

/// The latest preview of a picture, which is 404 until its first step.
async fn latest_preview(
    State(server): State<Arc<Server>>,
    Path(id): Path<String>,
) -> Result<Response, Failure> {
    let step = {
        let jobs = server.jobs.lock().expect("the jobs");
        jobs.get(&id).ok_or_else(Failure::unknown)?.preview_step
    };
    let step = step.ok_or_else(|| Failure(StatusCode::NOT_FOUND, "no preview yet".to_owned()))?;
    preview(State(server), Path((id, step))).await
}

/// The preview of a picture after step `step`, counted from 1.
async fn preview(
    State(server): State<Arc<Server>>,
    Path((id, step)): Path<(String, usize)>,
) -> Result<Response, Failure> {
    let directory = {
        let jobs = server.jobs.lock().expect("the jobs");
        jobs.get(&id)
            .ok_or_else(Failure::unknown)?
            .directory
            .clone()
    };
    let path = directory
        .join(PREVIEWS)
        .join(crate::image::preview_name(step));
    match tokio::fs::read(&path).await {
        Ok(bytes) => Ok(([(header::CONTENT_TYPE, "image/png")], bytes).into_response()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Err(Failure(
            StatusCode::NOT_FOUND,
            format!("no preview of step {step}"),
        )),
        Err(error) => Err(error.into()),
    }
}

/// What generation `id` wrote, when it is `output` and it has written it.
async fn written(
    server: &Server,
    id: &str,
    output: Output,
    content_type: &'static str,
) -> Result<Response, Failure> {
    let (state, path, wrote) = {
        let jobs = server.jobs.lock().expect("the jobs");
        let job = jobs.get(id).ok_or_else(Failure::unknown)?;
        (job.state, job.written.clone(), job.output)
    };
    if wrote != output {
        return Err(Failure(
            StatusCode::NOT_FOUND,
            match wrote {
                Output::Video => "that generation writes a video".to_owned(),
                Output::Picture => "that generation writes a picture".to_owned(),
            },
        ));
    }
    match state {
        Stage::Done => Ok((
            [(header::CONTENT_TYPE, content_type)],
            tokio::fs::read(&path).await?,
        )
            .into_response()),
        Stage::Failed => Err(Failure::gone("that generation failed")),
        Stage::Cancelled => Err(Failure::gone("that generation was cancelled")),
        _ => Err(Failure::waiting()),
    }
}

/// What the value of a request's field is.
#[derive(Clone, Copy)]
enum FieldKind {
    Text,
    /// A file the request uploads.
    File,
    /// The name of a file in one of these directories of the models directory.
    ModelFile(&'static [&'static str]),
}

/// What kind of value the field `name` of a request takes, or why a request may not give it.
fn field_kind(name: &str) -> Result<FieldKind, String> {
    if !crate::generation::OPTIONS.contains(&name) {
        return Err(format!("{name} is not an option of a generation"));
    }
    if MACHINE_OPTIONS.contains(&name) {
        return Err(format!("{name} is the server's to set, not a request's"));
    }
    if FILE_OPTIONS.contains(&name) {
        return Ok(FieldKind::File);
    }
    Ok(MODEL_FILE_OPTIONS
        .iter()
        .find(|(option, _)| *option == name)
        .map_or(FieldKind::Text, |(_, directories)| {
            FieldKind::ModelFile(directories)
        }))
}

/// What kind of value the field `name` of a request for a picture takes, or why it may not give it.
fn image_field_kind(name: &str) -> Result<FieldKind, String> {
    if !IMAGE_OPTIONS.contains(&name) {
        return Err(format!("{name} is not an option of a picture"));
    }
    if IMAGE_FILE_OPTIONS.contains(&name) {
        return Ok(FieldKind::File);
    }
    Ok(MODEL_FILE_OPTIONS
        .iter()
        .find(|(option, _)| *option == name)
        .map_or(FieldKind::Text, |(_, directories)| {
            FieldKind::ModelFile(directories)
        }))
}

/// The path of the file a request's `option` names, the first of `directories` in the models
/// directory that has it. A name is only a name, so one with a directory in it is refused rather
/// than followed out of the models directory.
fn model_named(
    models: Option<&std::path::Path>,
    option: &str,
    name: &str,
    directories: &[&str],
) -> Result<String, String> {
    let is_name = std::path::Path::new(name).file_name() == Some(name.as_ref());
    if !is_name || name.starts_with('.') {
        return Err(format!("{option} takes the name of a file, not {name}"));
    }
    let models =
        models.ok_or_else(|| format!("this server has no models directory for {option}"))?;
    directories
        .iter()
        .map(|directory| models.join(directory).join(name))
        .find(|path| path.is_file())
        .map(|path| path.to_string_lossy().into_owned())
        .ok_or_else(|| {
            format!(
                "{option} {name}: none in the models directory's {}",
                directories.join(" or ")
            )
        })
}

/// Where an upload of field `field` named `name` is written: a reference into the job's
/// `references` directory, numbered when two references share a name, and anything else beside
/// the job.
async fn upload_path(
    directory: &std::path::Path,
    field: &str,
    name: &str,
) -> Result<PathBuf, Failure> {
    if field != "reference" {
        return Ok(directory.join(name));
    }
    let references = directory.join(REFERENCES);
    tokio::fs::create_dir_all(&references).await?;
    let mut path = references.join(name);
    let mut count = 1;
    while tokio::fs::try_exists(&path).await? {
        count += 1;
        path = references.join(format!("{count}-{name}"));
    }
    Ok(path)
}

/// The file name a client gave, taken down to a name. A name with a directory in it is a name
/// meant to land somewhere other than where it was sent.
fn named(filename: &str) -> Result<String, Failure> {
    let name = std::path::Path::new(filename)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    match name.is_empty() || name.starts_with('.') {
        true => Err(Failure::asked("a file name that names no file")),
        false => Ok(name.to_owned()),
    }
}

/// An id no other job of this server has. The clock says which run of the server it belongs to
/// and the count says which job of that run, so a client holding an id from a server that has
/// been restarted is told there is no such generation rather than handed somebody else's.
fn next_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNT: AtomicU64 = AtomicU64::new(0);
    let started = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!(
        "{:x}{:04x}",
        started.as_secs(),
        COUNT.fetch_add(1, Ordering::Relaxed) & 0xffff
    )
}

/// What a request is answered with when it is not answered with what it asked for.
struct Failure(StatusCode, String);

impl Failure {
    fn asked(message: &str) -> Self {
        Failure(StatusCode::BAD_REQUEST, message.to_owned())
    }

    fn form(error: axum::extract::multipart::MultipartError) -> Self {
        Failure(StatusCode::BAD_REQUEST, error.to_string())
    }

    fn unknown() -> Self {
        Failure(StatusCode::NOT_FOUND, "no such generation".to_owned())
    }

    fn waiting() -> Self {
        Failure(
            StatusCode::CONFLICT,
            "that generation has not written anything yet".to_owned(),
        )
    }

    fn gone(message: &str) -> Self {
        Failure(StatusCode::CONFLICT, message.to_owned())
    }
}

impl From<std::io::Error> for Failure {
    fn from(error: std::io::Error) -> Self {
        Failure(StatusCode::INTERNAL_SERVER_ERROR, error.to_string())
    }
}

impl IntoResponse for Failure {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "error": self.1 }))).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Held by a test that runs a job, since the run's cancellation flag is the process's.
    static RUNS: Mutex<()> = Mutex::new(());

    /// A server with no generating thread, whose queue is handed back for a test to run.
    fn test_server(name: &str) -> (Arc<Server>, std::sync::mpsc::Receiver<String>) {
        let directory =
            std::env::temp_dir().join(format!("mmh3-server-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let (queue, waiting) = channel();
        let server = Arc::new(Server {
            jobs: Mutex::new(HashMap::new()),
            queue,
            defaults: Vec::new(),
            image_defaults: Vec::new(),
            models: None,
            ffmpeg: None,
            container: Container::Mp4,
            directory,
            announcements: tokio::sync::broadcast::channel(ANNOUNCEMENTS).0,
        });
        (server, waiting)
    }

    fn block<F: std::future::Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(future)
    }

    /// The status and the JSON body of an answer.
    fn answer(response: Result<Response, Failure>) -> (StatusCode, serde_json::Value) {
        let response = response.unwrap_or_else(IntoResponse::into_response);
        let status = response.status();
        let bytes = block(axum::body::to_bytes(response.into_body(), usize::MAX)).unwrap();
        (status, serde_json::from_slice(&bytes).unwrap_or_default())
    }

    fn queue_picture(server: &Server) -> String {
        queue_picture_with(server, &[])
    }

    fn queue_picture_with(server: &Server, arguments: &[&str]) -> String {
        let id = next_id();
        let directory = server.directory.join(&id);
        std::fs::create_dir_all(&directory).unwrap();
        let written = directory.join("image.png");
        let arguments = arguments
            .iter()
            .map(|&argument| argument.to_owned())
            .collect();
        queue_job(
            server,
            id.clone(),
            Output::Picture,
            arguments,
            directory,
            written,
        )
        .map_err(|_| "queued")
        .unwrap();
        id
    }

    fn state(server: &Server, id: &str) -> &'static str {
        server.jobs.lock().unwrap()[id].state.name()
    }

    #[test]
    fn a_cancelled_generation_stops_where_it_is() {
        let _runs = RUNS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (server, waiting) = test_server("cancel");
        let running = queue_picture(&server);
        let queued = queue_picture(&server);

        // One waiting its turn is cancelled at once and never runs.
        let (status, body) = answer(block(cancel(
            State(Arc::clone(&server)),
            Path(queued.clone()),
        )));
        assert_eq!(status, StatusCode::ACCEPTED);
        assert_eq!(body["state"], "cancelled");

        // One running stops at the next step boundary it reaches.
        assert_eq!(waiting.recv().unwrap(), running);
        generate(&server, &running, |_, _, _| {
            assert_eq!(state(&server, &running), "running");
            crate::generation::check_cancelled()?;
            let (status, body) = answer(block(cancel(
                State(Arc::clone(&server)),
                Path(running.clone()),
            )));
            assert_eq!(status, StatusCode::ACCEPTED);
            assert_eq!(body["state"], "running");
            crate::generation::check_cancelled()?;
            unreachable!("the run was cancelled");
        });
        assert_eq!(state(&server, &running), "cancelled");

        assert_eq!(waiting.recv().unwrap(), queued);
        generate(&server, &queued, |_, _, _| {
            unreachable!("a cancelled job runs")
        });
        assert_eq!(state(&server, &queued), "cancelled");

        // The next run starts with the flag down, and one that has ended cannot be cancelled.
        let done = queue_picture(&server);
        generate(&server, &done, |_, _, _| {
            crate::generation::check_cancelled()?;
            Ok(())
        });
        assert_eq!(state(&server, &done), "done");
        let (status, _) = answer(block(cancel(
            State(Arc::clone(&server)),
            Path(done.clone()),
        )));
        assert_eq!(status, StatusCode::CONFLICT);
        let (status, _) = answer(block(cancel(
            State(Arc::clone(&server)),
            Path("nothing".to_owned()),
        )));
        assert_eq!(status, StatusCode::NOT_FOUND);
        std::fs::remove_dir_all(&server.directory).unwrap();
    }

    #[test]
    fn a_generation_says_what_was_asked_and_how_it_went() {
        let _runs = RUNS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (server, _waiting) = test_server("shape");
        let id = queue_picture_with(
            &server,
            &[
                "--out",
                "image.png",
                "--prompt",
                "a cat",
                "--width",
                "1024",
                "--reference",
                "/jobs/x/references/cat.png",
                "--reference",
                "/jobs/x/references/2-cat.png",
                "--attention",
                "sol",
                "--sparse-tau",
                "1.5",
                "--draft-steps",
                "2",
                "--latent-upscaler",
                "/models/latent_upscale_models/upscaler.safetensors",
            ],
        );
        let describe = || {
            answer(block(describe(
                State(Arc::clone(&server)),
                Path(id.clone()),
            )))
        };
        let (status, queued) = describe();
        assert_eq!(status, StatusCode::OK);
        assert_eq!(queued["kind"], "image");
        assert_eq!(queued["state"], "queued");
        assert_eq!(
            queued["request"],
            json!({
                "prompt": "a cat",
                "width": 1024,
                "height": 2048,
                "seed": 0,
                "min_generation_area": 4194304,
                "attention_precision": null,
                "attention": "sol",
                "sparse_tau": 1.5,
                "sparse_start": null,
                "draft_steps": 2,
                "latent_upscaler": "upscaler.safetensors",
                "renoise_sigma": null,
                "steps": null,
                "shift": null,
                "patch": null,
                "lora": null,
                "lora_strength": null,
                "lora_mode": null,
                "references": ["cat.png", "2-cat.png"],
            })
        );
        assert!(queued["created_at"].is_f64());
        for field in ["started_at", "finished_at", "timings", "preview_step"] {
            assert!(queued[field].is_null(), "{field}");
        }
        let latest = || {
            answer(block(latest_preview(
                State(Arc::clone(&server)),
                Path(id.clone()),
            )))
        };
        assert_eq!(latest().0, StatusCode::NOT_FOUND);

        let previews = server.directory.join(&id).join(PREVIEWS);
        generate(&server, &id, |_, _, watch| {
            std::fs::create_dir_all(&previews)?;
            std::fs::write(previews.join(crate::image::preview_name(1)), b"png")?;
            watch(&crate::image::Report {
                timings: crate::image::Timings {
                    load_dit: Some(2.0),
                    steps: vec![1.5],
                    ..Default::default()
                },
                preview_step: Some(1),
                models: Vec::new(),
                attention_precision: Some("fp16"),
            });
            Ok(())
        });
        let (_, done) = describe();
        assert_eq!(done["state"], "done");
        assert_eq!(done["preview_step"], 1);
        assert_eq!(done["request"]["attention_precision"], "fp16");
        assert_eq!(done["timings"]["steps"], json!([1.5]));
        assert_eq!(done["timings"]["load_dit"], 2.0);
        assert!(done["timings"]["decode"].is_null());
        assert!(done["started_at"].as_f64() <= done["finished_at"].as_f64());

        let response = block(latest_preview(State(Arc::clone(&server)), Path(id.clone())))
            .unwrap_or_else(IntoResponse::into_response);
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[header::CONTENT_TYPE], "image/png");
        let bytes = block(axum::body::to_bytes(response.into_body(), usize::MAX)).unwrap();
        assert_eq!(&bytes[..], b"png");
        let (status, _) = answer(block(preview(
            State(Arc::clone(&server)),
            Path((id.clone(), 2)),
        )));
        assert_eq!(status, StatusCode::NOT_FOUND);
        std::fs::remove_dir_all(&server.directory).unwrap();
    }

    #[test]
    fn a_picture_says_which_few_step_lora_it_asked_for() {
        let arguments: Vec<String> = [
            "--prompt",
            "a cat",
            "--steps",
            "4",
            "--lora",
            "/models/loras/pdmd.safetensors",
            "--draft-steps",
            "1",
            "--renoise-sigma",
            "0.8",
        ]
        .map(str::to_owned)
        .into();
        let asked = request(&arguments, Output::Picture);
        assert_eq!(asked["steps"], 4);
        assert!(asked["shift"].is_null());
        assert_eq!(asked["renoise_sigma"], 0.8);
        assert!(asked["patch"].is_null());
        assert_eq!(asked["lora"], "pdmd.safetensors");
        assert_eq!(asked["lora_strength"], 1.0);
        assert_eq!(asked["lora_mode"], "adapter");
    }

    #[test]
    fn a_server_started_again_restores_the_generations_before() {
        let _runs = RUNS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (server, waiting) = test_server("restore");
        let done = queue_picture_with(&server, &["--prompt", "a dog"]);
        let cancelled = queue_picture(&server);
        let interrupted = queue_picture(&server);
        std::fs::create_dir_all(server.directory.join("not-a-job")).unwrap();

        assert_eq!(waiting.recv().unwrap(), done);
        generate(&server, &done, |_, _, watch| {
            watch(&crate::image::Report {
                timings: crate::image::Timings {
                    steps: vec![1.0, 2.0],
                    ..Default::default()
                },
                preview_step: Some(2),
                ..Default::default()
            });
            Ok(())
        });
        block(cancel(State(Arc::clone(&server)), Path(cancelled.clone())))
            .unwrap_or_else(|_| unreachable!());
        assert_eq!(waiting.recv().unwrap(), cancelled);
        generate(&server, &cancelled, |_, _, _| unreachable!());
        // A server that starts while this one runs a job finds it running, as one that starts
        // after this one stopped half way through would.
        let mut during = None;
        assert_eq!(waiting.recv().unwrap(), interrupted);
        generate(&server, &interrupted, |_, _, _| {
            during = Some(restore_all(&server.directory));
            Ok(())
        });
        let during = during.unwrap();
        assert_eq!(during[&interrupted].state.name(), "failed");
        assert_eq!(during[&interrupted].failure.as_deref(), Some(INTERRUPTED));

        let restored = restore_all(&server.directory);
        let mut ids: Vec<&String> = restored.keys().collect();
        ids.sort();
        assert_eq!(ids, [&done, &cancelled, &interrupted]);
        let jobs = server.jobs.lock().unwrap();
        for id in [&done, &cancelled, &interrupted] {
            let (mut live, mut again) = (jobs[id].describe(id), restored[id].describe(id));
            for field in ["created_at", "started_at", "finished_at"] {
                let (one, other) = (live[field].as_f64(), again[field].as_f64());
                assert_eq!(one.is_some(), other.is_some(), "{id} {field}");
                assert!((one.unwrap_or(0.0) - other.unwrap_or(0.0)).abs() < 1e-3);
                live[field] = serde_json::Value::Null;
                again[field] = serde_json::Value::Null;
            }
            assert_eq!(live, again, "{id}");
        }
        drop(jobs);
        assert_eq!(
            restored[&done].describe(&done)["timings"]["steps"],
            json!([1.0, 2.0])
        );
        assert_eq!(restored[&done].preview_step, Some(2));
        assert_eq!(restored[&cancelled].state.name(), "cancelled");

        let metadata: serde_json::Value = serde_json::from_slice(
            &std::fs::read(server.directory.join(&done).join(METADATA)).unwrap(),
        )
        .unwrap();
        assert_eq!(metadata["mmh3"]["version"], env!("CARGO_PKG_VERSION"));
        assert!(metadata["models"].is_array());
        assert!(metadata["machine"].is_object());

        // The list of a server started again is the most recent first.
        let (again, _) = test_server("restore-list");
        *again.jobs.lock().unwrap() = restored;
        let response = block(list(State(Arc::clone(&again))));
        let (_, listed) = answer(Ok(response));
        let listed: Vec<&str> = listed["generations"]
            .as_array()
            .unwrap()
            .iter()
            .map(|generation| generation["id"].as_str().unwrap())
            .collect();
        assert_eq!(listed, [&interrupted, &cancelled, &done]);
        std::fs::remove_dir_all(&server.directory).unwrap();
        std::fs::remove_dir_all(&again.directory).unwrap();
    }

    #[test]
    fn one_stream_says_what_changes_of_every_generation() {
        let (server, _waiting) = test_server("events");
        let mut said = HashMap::new();
        assert!(changes(&server, &mut said).is_empty());
        let id = queue_picture(&server);
        let announced = changes(&server, &mut said);
        assert_eq!(announced.len(), 1);
        assert_eq!(announced[0].0, "generation");
        assert_eq!(announced[0].1["id"], id.as_str());
        assert_eq!(announced[0].1["state"], "queued");
        assert!(changes(&server, &mut said).is_empty());

        let report = crate::image::Report {
            preview_step: Some(2),
            ..Default::default()
        };
        server
            .jobs
            .lock()
            .unwrap()
            .get_mut(&id)
            .unwrap()
            .take_report(&report);
        let announced = changes(&server, &mut said);
        let names: Vec<&str> = announced.iter().map(|(name, _)| *name).collect();
        assert_eq!(names, ["generation", "preview"]);
        assert_eq!(announced[1].1["id"], id.as_str());
        assert_eq!(announced[1].1["step"], 2);
        assert_eq!(
            announced[1].1["url"],
            format!("/v1/generations/{id}/preview/2")
        );
        // A change that brings no new preview says only the generation.
        server.jobs.lock().unwrap().get_mut(&id).unwrap().state = Stage::Cancelled;
        let names: Vec<&str> = changes(&server, &mut said)
            .iter()
            .map(|(name, _)| *name)
            .collect();
        assert_eq!(names, ["generation"]);
        server.jobs.lock().unwrap().remove(&id);
        assert!(changes(&server, &mut said).is_empty());
        assert!(said.is_empty());

        // The stream starts with a snapshot and goes on with what is announced.
        let again = queue_picture(&server);
        let body = block(async {
            use futures_util::StreamExt;

            let response = all_events(State(Arc::clone(&server))).await.into_response();
            let mut frames = response.into_body().into_data_stream();
            let first = frames.next().await.unwrap().unwrap();
            server
                .announcements
                .send(("generation", json!({ "id": again })))
                .unwrap();
            let second = frames.next().await.unwrap().unwrap();
            [first, second].map(|frame| String::from_utf8(frame.to_vec()).unwrap())
        });
        assert!(
            body[0].starts_with("event: snapshot\ndata: {\"generations\":[{"),
            "{}",
            body[0]
        );
        assert!(body[0].contains(&again));
        assert_eq!(
            body[1],
            format!("event: generation\ndata: {{\"id\":\"{again}\"}}\n\n")
        );
        std::fs::remove_dir_all(&server.directory).unwrap();
    }

    #[test]
    fn references_are_uploaded_into_their_own_directory() {
        let directory = std::env::temp_dir().join(format!("mmh3-uploads-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let path = |field: &str, name: &str| {
            block(upload_path(&directory, field, name)).unwrap_or_else(|_| unreachable!())
        };
        let first = path("reference", "cat.png");
        assert_eq!(first, directory.join("references/cat.png"));
        std::fs::write(&first, b"").unwrap();
        assert_eq!(
            path("reference", "cat.png"),
            directory.join("references/2-cat.png")
        );
        assert_eq!(
            path("prompt-file", "prompt.txt"),
            directory.join("prompt.txt")
        );
        std::fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn a_request_gives_what_to_generate_and_not_the_machine() {
        assert!(matches!(field_kind("steps"), Ok(FieldKind::Text)));
        assert!(matches!(field_kind("reference"), Ok(FieldKind::File)));
        assert!(matches!(
            field_kind("lora"),
            Ok(FieldKind::ModelFile(["loras"]))
        ));
        for name in MACHINE_OPTIONS
            .iter()
            .chain(&["out", "local-worker", "listen"])
        {
            assert!(field_kind(name).is_err(), "{name}");
        }
    }

    #[test]
    fn a_request_for_a_picture_gives_what_to_generate() {
        assert!(matches!(image_field_kind("prompt"), Ok(FieldKind::Text)));
        assert!(matches!(image_field_kind("reference"), Ok(FieldKind::File)));
        assert!(matches!(
            image_field_kind("draft-steps"),
            Ok(FieldKind::Text)
        ));
        assert!(matches!(
            image_field_kind("latent-upscaler"),
            Ok(FieldKind::ModelFile(["latent_upscale_models"]))
        ));
        assert!(matches!(
            image_field_kind("lora"),
            Ok(FieldKind::ModelFile(["loras"]))
        ));
        for name in [
            "dit",
            "vae",
            "models",
            "out",
            "save-latent",
            "linear-precision",
            "sampler",
        ] {
            assert!(image_field_kind(name).is_err(), "{name}");
        }
    }

    #[test]
    fn a_model_file_is_a_name_in_the_models_directory() {
        let models = std::env::temp_dir().join(format!("mmh3-server-{}", std::process::id()));
        for directory in ["patches", "loras"] {
            std::fs::create_dir_all(models.join(directory)).unwrap();
        }
        std::fs::write(models.join("loras/both.safetensors"), b"").unwrap();
        std::fs::write(models.join("patches/both.safetensors"), b"").unwrap();
        let found = |name: &str, directories: &[&str]| {
            model_named(Some(&models), "lora", name, directories)
        };
        assert_eq!(
            found("both.safetensors", &["patches", "loras"]).unwrap(),
            models.join("patches/both.safetensors").to_string_lossy()
        );
        for name in [
            "missing.safetensors",
            "../loras/both.safetensors",
            "loras/both.safetensors",
            "/etc/hostname",
            ".hidden",
            "",
        ] {
            assert!(found(name, &["loras"]).is_err(), "{name}");
        }
        assert!(model_named(None, "lora", "both.safetensors", &["loras"]).is_err());
        std::fs::remove_dir_all(&models).unwrap();
    }
}

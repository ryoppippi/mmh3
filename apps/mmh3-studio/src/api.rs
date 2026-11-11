//! A blocking client for `mmh3 server` (`docs/server.md` in mmh3, plus the image additions in
//! the studio contract). Every call blocks, so run them off the UI thread.

use std::io::{BufRead, BufReader, Read};
use std::time::Duration;

use serde::{Deserialize, Serialize};

pub const DEFAULT_SERVER: &str = "http://127.0.0.1:8833";
pub const IMAGE_STEPS: u32 = 6;
const MAX_PNG_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum State {
    Queued,
    Running,
    Done,
    Failed,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Image,
    #[default]
    Video,
    #[serde(other)]
    Other,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Timings {
    pub text: f64,
    pub steps: Vec<f64>,
    pub decode: f64,
}

impl Timings {
    pub fn steps_total(&self) -> f64 {
        self.steps.iter().sum()
    }

    pub fn total(&self) -> f64 {
        self.text + self.steps_total() + self.decode
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Generation {
    pub id: String,
    pub state: State,
    pub queued: u64,
    pub seconds: Option<f64>,
    pub step: Option<u32>,
    pub steps: Option<u32>,
    pub error: Option<String>,
    #[serde(default)]
    pub kind: Kind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timings: Option<Timings>,
}

impl Generation {
    pub fn is_finished(&self) -> bool {
        matches!(self.state, State::Done | State::Failed)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Memory {
    pub free: u64,
    pub total: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Status {
    pub generating: Option<String>,
    pub generations: u32,
    #[serde(default)]
    pub memory: Option<Memory>,
    pub models: Option<Vec<String>>,
    pub queued: u32,
}

/// What the studio asks for: an image generation of the server's fixed six steps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageRequest {
    pub prompt: String,
    pub seed: u64,
    pub width: u32,
    pub height: u32,
    pub min_generation_area: Option<u64>,
}

impl ImageRequest {
    pub fn fields(&self) -> Vec<(&'static str, String)> {
        let mut fields = vec![
            ("image", "1".to_owned()),
            ("prompt", self.prompt.clone()),
            ("seed", self.seed.to_string()),
            ("width", self.width.to_string()),
            ("height", self.height.to_string()),
        ];
        if let Some(area) = self.min_generation_area {
            fields.push(("min-generation-area", area.to_string()));
        }
        fields
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// Nothing answered: the server isn't running, or the address is wrong.
    Unreachable(String),
    /// The server answered with an error status; its body says why.
    Refused { status: u16, message: String },
    /// The answer wasn't what the contract says.
    Malformed(String),
}

impl Error {
    pub fn is_unreachable(&self) -> bool {
        matches!(self, Error::Unreachable(_))
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Unreachable(why) => write!(f, "The server didn't answer: {why}"),
            Error::Refused { status, message } if message.is_empty() => write!(f, "The server refused it (HTTP {status})."),
            Error::Refused { status, message } => write!(f, "{message} (HTTP {status})"),
            Error::Malformed(why) => write!(f, "The server's answer couldn't be read: {why}"),
        }
    }
}

impl std::error::Error for Error {}

/// Builds a `multipart/form-data` body of text fields; returns it with its content type.
pub fn multipart(fields: &[(&str, String)], boundary: &str) -> (String, Vec<u8>) {
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes());
        body.extend_from_slice(value.as_bytes());
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

/// Reads server-sent events line by line: `data:` lines gather until a blank line ends the event.
#[derive(Default)]
pub struct SseParser {
    data: Vec<String>,
}

impl SseParser {
    pub fn line(&mut self, line: &str) -> Option<String> {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.is_empty() {
            if self.data.is_empty() {
                return None;
            }
            return Some(std::mem::take(&mut self.data).join("\n"));
        }
        if let Some(rest) = line.strip_prefix("data:") {
            self.data.push(rest.strip_prefix(' ').unwrap_or(rest).to_owned());
        }
        None
    }

    pub fn finish(&mut self) -> Option<String> {
        self.line("")
    }
}

#[derive(Clone)]
pub struct Client {
    base: String,
    quick: ureq::Agent,
    long: ureq::Agent,
}

impl Client {
    pub fn new(base: impl Into<String>) -> Client {
        let base = base.into().trim_end_matches('/').to_owned();
        let agent = |global: Option<Duration>| -> ureq::Agent {
            ureq::Agent::config_builder().http_status_as_error(false).timeout_connect(Some(Duration::from_secs(2))).timeout_global(global).build().into()
        };
        Client { base, quick: agent(Some(Duration::from_secs(5))), long: agent(None) }
    }

    pub fn base(&self) -> &str {
        &self.base
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }

    pub fn status(&self) -> Result<Status, Error> {
        json(self.quick.get(&self.url("/v1/status")).call())
    }

    pub fn list(&self) -> Result<Vec<Generation>, Error> {
        json(self.quick.get(&self.url("/v1/generations")).call())
    }

    pub fn generation(&self, id: &str) -> Result<Generation, Error> {
        json(self.quick.get(&self.url(&format!("/v1/generations/{id}"))).call())
    }

    pub fn create_image(&self, request: &ImageRequest) -> Result<Generation, Error> {
        let boundary = format!("mmh3-studio-{:016x}", request.seed ^ 0x9e37_79b9_7f4a_7c15);
        let (content_type, body) = multipart(&request.fields(), &boundary);
        json(self.quick.post(&self.url("/v1/generations")).header("Content-Type", &content_type).send(&body[..]))
    }

    pub fn delete(&self, id: &str) -> Result<(), Error> {
        checked(self.quick.delete(&self.url(&format!("/v1/generations/{id}"))).call()).map(drop)
    }

    pub fn image(&self, id: &str) -> Result<Vec<u8>, Error> {
        let mut response = checked(self.long.get(&self.url(&format!("/v1/generations/{id}/image"))).call())?;
        response.body_mut().with_config().limit(MAX_PNG_BYTES).read_to_vec().map_err(|e| Error::Unreachable(e.to_string()))
    }

    /// Follows a generation's events until the server closes the stream, handing each to `each`.
    pub fn events(&self, id: &str, each: impl FnMut(Generation)) -> Result<(), Error> {
        let response = checked(self.long.get(&self.url(&format!("/v1/generations/{id}/events"))).header("Accept", "text/event-stream").call())?;
        read_events(response.into_body().into_reader(), each)
    }
}

pub fn read_events(reader: impl Read, mut each: impl FnMut(Generation)) -> Result<(), Error> {
    let mut parser = SseParser::default();
    let mut emit = |data: String| -> Result<(), Error> {
        let g = serde_json::from_str(&data).map_err(|e| Error::Malformed(e.to_string()))?;
        each(g);
        Ok(())
    };
    for line in BufReader::new(reader).lines() {
        let line = line.map_err(|e| Error::Unreachable(e.to_string()))?;
        if let Some(data) = parser.line(&line) {
            emit(data)?;
        }
    }
    if let Some(data) = parser.finish() {
        emit(data)?;
    }
    Ok(())
}

type Answer = Result<ureq::http::Response<ureq::Body>, ureq::Error>;

fn checked(answer: Answer) -> Result<ureq::http::Response<ureq::Body>, Error> {
    let mut response = answer.map_err(|e| Error::Unreachable(e.to_string()))?;
    let status = response.status().as_u16();
    if (200..300).contains(&status) {
        return Ok(response);
    }
    let message = response.body_mut().read_to_string().unwrap_or_default().trim().to_owned();
    Err(Error::Refused { status, message })
}

fn json<T: serde::de::DeserializeOwned>(answer: Answer) -> Result<T, Error> {
    let mut response = checked(answer)?;
    let text = response.body_mut().read_to_string().map_err(|e| Error::Unreachable(e.to_string()))?;
    serde_json::from_str(&text).map_err(|e| Error::Malformed(e.to_string()))
}

#[cfg(test)]
mod tests;

use super::*;

const QUEUED_VIDEO: &str = include_str!("../../fixtures/generation-queued-video.json");
const DONE_IMAGE: &str = include_str!("../../fixtures/generation-done-image.json");
const LIST: &str = include_str!("../../fixtures/generations.json");
const STATUS_IDLE: &str = include_str!("../../fixtures/status-idle.json");
const STATUS_BUSY: &str = include_str!("../../fixtures/status-busy.json");
const EVENTS_VIDEO: &str = include_str!("../../fixtures/events-video.txt");
const EVENTS_IMAGE: &str = include_str!("../../fixtures/events-image.txt");

#[test]
fn a_video_generation_from_an_older_server_reads_as_a_video() {
    let g: Generation = serde_json::from_str(QUEUED_VIDEO).unwrap();
    assert_eq!(g.id, "6ab1cf3b0000");
    assert_eq!(g.state, State::Queued);
    assert_eq!(g.kind, Kind::Video);
    assert_eq!((g.width, g.prompt.as_deref(), g.timings.as_ref()), (None, None, None));
}

#[test]
fn a_finished_image_carries_its_request_and_timings() {
    let g: Generation = serde_json::from_str(DONE_IMAGE).unwrap();
    assert_eq!(g.kind, Kind::Image);
    assert_eq!(g.state, State::Done);
    assert_eq!((g.width, g.height, g.seed), (Some(5472), Some(3072), Some(4127)));
    assert_eq!(g.prompt.as_deref(), Some("A lighthouse on a basalt cliff at dusk, long exposure"));
    let t = g.timings.unwrap();
    assert_eq!(t.steps.len(), 6);
    assert!((t.steps_total() - 11.85).abs() < 1e-9);
    assert!((t.total() - (0.46 + 11.85 + 2.31)).abs() < 1e-9);
}

#[test]
fn the_list_mixes_states_and_kinds() {
    let list: Vec<Generation> = serde_json::from_str(LIST).unwrap();
    let states: Vec<State> = list.iter().map(|g| g.state).collect();
    assert_eq!(states, [State::Running, State::Failed, State::Done, State::Done]);
    assert_eq!(list[0].step, Some(3));
    assert_eq!(list[1].error.as_deref(), Some("width must be a multiple of 32"));
    assert_eq!(list.iter().filter(|g| g.kind == Kind::Image).count(), 3);
}

#[test]
fn an_unknown_kind_does_not_break_the_list() {
    let g: Generation = serde_json::from_str(&DONE_IMAGE.replace("\"image\"", "\"audio\"")).unwrap();
    assert_eq!(g.kind, Kind::Other);
}

#[test]
fn status_reads_models_or_their_absence_while_busy() {
    let idle: Status = serde_json::from_str(STATUS_IDLE).unwrap();
    assert_eq!(idle.models.as_deref(), Some(&["text_encoder".to_owned(), "dit".to_owned(), "image_vae".to_owned()][..]));
    assert_eq!(idle.memory, Some(Memory { free: 57_481_990_144, total: 130_660_909_056 }));
    let busy: Status = serde_json::from_str(STATUS_BUSY).unwrap();
    assert_eq!(busy.models, None);
    assert_eq!(busy.generating.as_deref(), Some("6ab1cf3b0000"));
    assert_eq!(busy.queued, 1);
}

#[test]
fn status_reads_without_memory_on_a_server_that_cannot_measure_it() {
    let json = STATUS_IDLE.replace(r#"{"free": 57481990144, "total": 130660909056}"#, "null");
    let status: Status = serde_json::from_str(&json).unwrap();
    assert_eq!(status.memory, None);
    let without: Status = serde_json::from_str(r#"{"generating": null, "generations": 0, "models": [], "queued": 0}"#).unwrap();
    assert_eq!(without.memory, None);
}

#[test]
fn events_from_the_docs_parse_one_per_step_then_the_end() {
    let mut seen = Vec::new();
    read_events(EVENTS_VIDEO.as_bytes(), |g| seen.push((g.state, g.step))).unwrap();
    assert_eq!(seen, [(State::Running, Some(3)), (State::Done, None)]);
}

#[test]
fn image_events_with_crlf_and_comments_parse_six_steps_and_timings() {
    let mut seen = Vec::new();
    read_events(EVENTS_IMAGE.as_bytes(), |g| seen.push(g)).unwrap();
    assert_eq!(seen.len(), 7);
    assert_eq!(seen.iter().filter_map(|g| g.step).collect::<Vec<_>>(), [1, 2, 3, 4, 5, 6]);
    assert!(seen[..6].iter().all(|g| g.steps == Some(IMAGE_STEPS)));
    assert_eq!(seen[6].state, State::Done);
    assert_eq!(seen[6].timings.as_ref().map(|t| t.steps.len()), Some(6));
}

#[test]
fn an_event_split_over_data_lines_joins_with_newlines() {
    let mut p = SseParser::default();
    assert_eq!(p.line("data: {\"a\":"), None);
    assert_eq!(p.line("data:1}"), None);
    assert_eq!(p.line("event: ignored"), None);
    assert_eq!(p.line(""), Some("{\"a\":\n1}".to_owned()));
    assert_eq!(p.line(""), None);
    assert_eq!(p.line("data: tail"), None);
    assert_eq!(p.finish(), Some("tail".to_owned()));
}

#[test]
fn the_request_names_the_image_fields() {
    let r = ImageRequest { prompt: "a fox".into(), seed: 42, width: 2048, height: 1024, min_generation_area: None };
    let names: Vec<&str> = r.fields().iter().map(|(n, _)| *n).collect();
    assert_eq!(names, ["image", "prompt", "seed", "width", "height"]);
    let with_area = ImageRequest { min_generation_area: Some(4_194_304), ..r };
    assert_eq!(with_area.fields().last(), Some(&("min-generation-area", "4194304".to_owned())));
}

#[test]
fn multipart_bodies_frame_each_field() {
    let (content_type, body) = multipart(&[("prompt", "a fox\nin snow".into()), ("seed", "7".into())], "B");
    assert_eq!(content_type, "multipart/form-data; boundary=B");
    assert_eq!(
        String::from_utf8(body).unwrap(),
        "--B\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\na fox\nin snow\r\n\
         --B\r\nContent-Disposition: form-data; name=\"seed\"\r\n\r\n7\r\n--B--\r\n"
    );
}

#[test]
fn errors_say_what_happened() {
    assert_eq!(Error::Refused { status: 409, message: "still running".into() }.to_string(), "still running (HTTP 409)");
    assert_eq!(Error::Refused { status: 500, message: String::new() }.to_string(), "The server refused it (HTTP 500).");
    assert!(Error::Unreachable("refused".into()).is_unreachable());
}

#[test]
fn an_unreachable_server_is_reported_as_such() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    let err = Client::new(format!("http://127.0.0.1:{port}")).status().unwrap_err();
    assert!(err.is_unreachable(), "{err:?}");
}

use mmh3_studio::api::{self, Generation, Kind, Timings};

use super::*;

fn online() -> State {
    let mut s = State::offline();
    s.conn = Conn::Connected(Status::default());
    s
}

fn server(id: &str, state: api::State) -> Generation {
    Generation {
        id: id.into(),
        state,
        queued: 1,
        seconds: None,
        step: None,
        steps: None,
        error: None,
        kind: Kind::Image,
        width: Some(2048),
        height: Some(2048),
        prompt: Some("a fox".into()),
        seed: Some(7),
        timings: None,
    }
}

fn running(id: &str, step: u32) -> Generation {
    Generation { step: Some(step), steps: Some(6), ..server(id, api::State::Running) }
}

#[test]
fn running_keeps_the_time_it_started_and_done_asks_for_the_picture() {
    let mut phase = Phase::Queued;
    assert!(step(&mut phase, &running("a", 1), 2.0).is_empty());
    assert_eq!(phase, Phase::Running { step: 1, steps: 6, since: 2.0 });
    let _ = step(&mut phase, &running("a", 4), 5.0);
    assert_eq!(phase, Phase::Running { step: 4, steps: 6, since: 2.0 });

    let t = Timings { text: 0.4, steps: vec![1.0; 6], decode: 2.0 };
    let done = Generation { seconds: Some(8.4), timings: Some(t.clone()), ..server("a", api::State::Done) };
    let fx: Vec<Effect> = step(&mut phase, &done, 9.0).into_iter().collect();
    assert_eq!(fx, [Effect::FetchImage]);
    assert_eq!(phase, Phase::Done { seconds: 8.4, timings: Some(t) });
    assert!(step(&mut phase, &done, 9.5).is_empty(), "a repeated end asks for nothing");
}

#[test]
fn an_accepted_submission_starts_watching() {
    let mut phase = Phase::Sending;
    let fx: Vec<Effect> = step(&mut phase, &server("a", api::State::Queued), 0.0).into_iter().collect();
    assert_eq!(fx, [Effect::Watch]);
}

#[test]
fn a_failure_keeps_its_reason() {
    let mut phase = Phase::Queued;
    let failed = Generation { error: Some("out of memory".into()), ..server("a", api::State::Failed) };
    let _ = step(&mut phase, &failed, 0.0);
    assert_eq!(phase, Phase::Failed("out of memory".into()));
}

#[test]
fn submitting_adds_a_selected_placeholder_that_takes_the_server_id() {
    let mut s = online();
    s.prompt.replace_text("a fox");
    s.generate();
    assert_eq!(s.gens.len(), 1);
    let local = s.gens[0].id.clone();
    assert!(local.starts_with("local-"));
    assert_eq!(s.selected.as_deref(), Some(local.as_str()));
    s.submitted(&local, Ok(server("srv1", api::State::Queued)), 1.0);
    assert_eq!(s.gens[0].id, "srv1");
    assert_eq!(s.gens[0].phase, Phase::Queued);
    assert_eq!(s.selected.as_deref(), Some("srv1"));
}

#[test]
fn a_refused_submission_fails_with_the_servers_words() {
    let mut s = online();
    s.prompt.replace_text("a fox");
    s.generate();
    let local = s.gens[0].id.clone();
    s.submitted(&local, Err(api::Error::Refused { status: 400, message: "width: bad".into() }), 1.0);
    assert_eq!(s.gens[0].phase, Phase::Failed("width: bad (HTTP 400)".into()));
}

#[test]
fn merging_the_list_keeps_local_placeholders_and_drops_videos_and_vanished_items() {
    let mut s = online();
    s.prompt.replace_text("pending");
    s.generate();
    s.merge(vec![running("b", 2), server("a", api::State::Done), Generation { kind: Kind::Video, ..server("v", api::State::Done) }], 3.0);
    let ids: Vec<&str> = s.gens.iter().map(|g| g.id.as_str()).collect();
    assert_eq!(ids, ["local-1", "b", "a"]);
    s.merge(vec![running("b", 3)], 4.0);
    let ids: Vec<&str> = s.gens.iter().map(|g| g.id.as_str()).collect();
    assert_eq!(ids, ["local-1", "b"]);
    assert_eq!(s.gens[1].phase, Phase::Running { step: 3, steps: 6, since: 3.0 });
}

#[test]
fn a_running_generation_is_not_deleted() {
    let mut s = State::offline();
    s.merge(vec![running("b", 2), server("a", api::State::Done)], 0.0);
    s.delete("b");
    assert_eq!(s.gens.len(), 2);
    assert_eq!(s.notes.len(), 1);
    assert_eq!(s.deleting, None);
}

#[test]
fn deleting_asks_first_and_cancelling_keeps_the_generation() {
    let mut s = State::offline();
    s.merge(vec![server("a", api::State::Done)], 0.0);
    s.delete("a");
    assert_eq!(s.deleting.as_deref(), Some("a"));
    assert_eq!(s.gens.len(), 1, "nothing is deleted before it is confirmed");
    s.deleting = None;
    s.confirm_delete();
    assert_eq!(s.gens.len(), 1);
    s.delete("a");
    s.confirm_delete();
    assert!(s.gens.is_empty());
    assert_eq!(s.deleting, None);
}

#[test]
fn a_failed_placeholder_that_never_reached_the_server_goes_at_once() {
    let mut s = online();
    s.prompt.replace_text("a fox");
    s.generate();
    let local = s.gens[0].id.clone();
    s.submitted(&local, Err(api::Error::Unreachable("refused".into())), 1.0);
    s.delete(&local);
    assert!(s.gens.is_empty());
    assert_eq!(s.deleting, None);
}

#[test]
fn generating_while_disconnected_adds_nothing_and_says_why() {
    let mut s = State::offline();
    s.conn = Conn::Disconnected("refused".into());
    s.prompt.replace_text("a fox");
    s.generate();
    assert!(s.gens.is_empty());
    assert_eq!(s.notes.last().map(|n| n.title.as_str()), Some("Start mmh3 server first."));
    s.merge(vec![server("a", api::State::Done)], 0.0);
    s.rerun("a");
    assert_eq!(s.gens.len(), 1);
}

#[test]
fn a_list_that_lands_before_the_submit_answer_leaves_one_row() {
    let mut s = online();
    s.prompt.replace_text("a fox");
    s.generate();
    let local = s.gens[0].id.clone();
    s.merge(vec![running("srv1", 1)], 1.0);
    assert_eq!(s.gens.len(), 2);
    s.submitted(&local, Ok(server("srv1", api::State::Queued)), 1.5);
    let ids: Vec<&str> = s.gens.iter().map(|g| g.id.as_str()).collect();
    assert_eq!(ids, ["srv1"]);
    assert_eq!(s.selected.as_deref(), Some("srv1"));
    assert_eq!(s.gens[0].phase, Phase::Running { step: 1, steps: 6, since: 1.0 }, "the newer state is kept");
}

#[test]
fn a_status_the_studio_cannot_read_still_counts_as_connected() {
    let mut s = State::offline();
    s.set_status(Err(api::Error::Malformed("missing field `queued`".into())));
    assert!(matches!(s.conn, Conn::Unreadable(_)));
    assert!(s.connected());
    s.set_status(Err(api::Error::Unreachable("refused".into())));
    assert!(matches!(s.conn, Conn::Disconnected(_)));
    assert!(!s.connected());
}

#[test]
fn generate_needs_a_prompt_and_a_valid_size_and_seed() {
    let mut s = State::offline();
    s.prompt.replace_text("a fox");
    assert_eq!(s.request().unwrap_err(), "Start mmh3 server first.");
    s = online();
    assert_eq!(s.request().unwrap_err(), "Write a prompt first.");
    s.prompt.replace_text("  a fox ");
    s.size = SizeChoice::Custom;
    s.custom_w.replace_text("1000");
    assert!(s.request().unwrap_err().contains("multiples of 32"));
    s.custom_w.replace_text("1024");
    s.seed.replace_text("abc");
    assert_eq!(s.request().unwrap_err(), "The seed must be a whole number.");
    s.seed.replace_text("42");
    let r = s.request().unwrap();
    assert_eq!((r.prompt.as_str(), r.width, r.height, r.seed), ("a fox", 1024, 1024, 42));
    s.seed.replace_text("");
    assert!(s.request().is_ok(), "an empty seed is a random one");
}

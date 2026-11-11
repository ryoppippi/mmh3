use std::time::Duration;

use mmh3_studio::api::{Client, Error, ImageRequest, Kind, State};
use mmh3_studio::mock::{self, Config};

fn quick_server() -> Client {
    let ms = Duration::from_millis;
    let addr = mock::spawn("127.0.0.1:0", Config { text: ms(20), step: ms(30), decode: ms(20) }).unwrap();
    Client::new(format!("http://{addr}"))
}

fn request(prompt: &str, seed: u64) -> ImageRequest {
    ImageRequest { prompt: prompt.into(), seed, width: 320, height: 256, min_generation_area: None }
}

#[test]
fn a_generation_streams_six_steps_then_serves_its_png() {
    let client = quick_server();
    let created = client.create_image(&request("a red fox in snow", 11)).unwrap();
    assert_eq!((created.state, created.kind), (State::Queued, Kind::Image));
    assert_eq!((created.width, created.height, created.seed), (Some(320), Some(256), Some(11)));

    let mut seen = Vec::new();
    client.events(&created.id, |g| seen.push(g)).unwrap();
    let steps: Vec<u32> = seen.iter().filter_map(|g| g.step).collect();
    assert_eq!(steps, [1, 2, 3, 4, 5, 6]);
    let last = seen.last().unwrap();
    assert_eq!(last.state, State::Done);
    assert_eq!(last.timings.as_ref().map(|t| t.steps.len()), Some(6));

    let png = client.image(&created.id).unwrap();
    let img = image::load_from_memory(&png).unwrap();
    assert_eq!((img.width(), img.height()), (320, 256));

    let list = client.list().unwrap();
    assert_eq!(list.len(), 1);
    assert_eq!(client.status().unwrap().models.map(|m| m.len()), Some(3));

    client.delete(&created.id).unwrap();
    assert!(client.list().unwrap().is_empty());
}

#[test]
fn the_server_refuses_what_the_contract_forbids() {
    let client = quick_server();
    let bad = client.create_image(&ImageRequest { width: 1000, ..request("x", 1) }).unwrap_err();
    assert!(matches!(bad, Error::Refused { status: 400, .. }), "{bad:?}");

    let g = client.create_image(&request("slow", 2)).unwrap();
    assert!(matches!(client.image(&g.id), Err(Error::Refused { status: 409, .. })));
    std::thread::sleep(Duration::from_millis(80));
    assert!(matches!(client.delete(&g.id), Err(Error::Refused { status: 409, .. })));
    assert!(matches!(client.generation("nope"), Err(Error::Refused { status: 404, .. })));
}

#[test]
fn a_failing_prompt_ends_failed_with_its_reason() {
    let client = quick_server();
    let g = client.create_image(&request("please [fail]", 3)).unwrap();
    let mut last = None;
    client.events(&g.id, |e| last = Some(e)).unwrap();
    let last = last.unwrap();
    assert_eq!(last.state, State::Failed);
    assert!(last.error.unwrap().contains("fail"));
}

#[test]
fn queued_generations_wait_their_turn() {
    let client = quick_server();
    let a = client.create_image(&request("first", 1)).unwrap();
    let b = client.create_image(&request("second", 2)).unwrap();
    std::thread::sleep(Duration::from_millis(40));
    let status = client.status().unwrap();
    assert_eq!(status.generating.as_deref(), Some(a.id.as_str()));
    assert_eq!(status.queued, 1);
    assert_eq!(status.models, None);
    let list = client.list().unwrap();
    assert_eq!(list[0].id, b.id, "most recent first");
}

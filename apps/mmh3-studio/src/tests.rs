use rune::{Env, Key, KeyEvent, Mods, Runtime};

use super::*;

fn runtime(scene: fn(&mut State), dark: bool) -> Runtime<State> {
    let mut s = State::offline();
    scene(&mut s);
    let mut rt = Runtime::new(s, view::view, (1320.0, 860.0), Env { system_dark: dark, scale: 2.0 });
    rt.poll = Some(poll);
    rt.frame(0.0);
    for _ in 0..30 {
        rt.frame(1.0 / 60.0);
    }
    rt
}

fn text_of(rt: &Runtime<State>, key: &str) -> String {
    rt.elements().into_iter().find(|e| e.key.as_deref() == Some(key)).and_then(|e| e.text).unwrap_or_default()
}

#[test]
fn every_scene_builds_in_both_appearances() {
    for (name, scene) in scenes::SCENES {
        for dark in [false, true] {
            let rt = runtime(*scene, dark);
            assert!(rt.find("status-bar").is_some(), "{name}: no status bar");
        }
    }
}

#[test]
fn the_disconnected_state_shows_how_to_start_the_server() {
    let rt = runtime(scenes::disconnected, false);
    assert!(rt.find("disconnected").is_some());
    assert!(rt.find("start-command").is_some());
    assert!(text_of(&rt, "status-connection").starts_with("Not connected"));
}

#[test]
fn cmd_return_in_the_prompt_generates_and_return_types_a_newline() {
    let mut rt = runtime(scenes::empty, false);
    let (x, y) = rt.find("prompt").unwrap();
    rt.pointer_down(x, y);
    rt.pointer_up(x, y);
    rt.frame(1.0 / 60.0);
    rt.update(|s, _| s.prompt.replace_text("a fox"));
    rt.frame(1.0 / 60.0);
    rt.key(KeyEvent { key: Key::Enter, mods: Mods::default() });
    rt.frame(1.0 / 60.0);
    assert_eq!(rt.state.prompt.text, "a fox\n");
    assert!(rt.state.gens.is_empty());
    rt.key(KeyEvent { key: Key::Enter, mods: Mods { cmd: true, ..Mods::default() } });
    rt.frame(1.0 / 60.0);
    assert_eq!(rt.state.gens.len(), 1);
    assert_eq!(rt.state.gens[0].prompt, "a fox");
}

#[test]
fn picking_a_size_tile_selects_it() {
    let mut rt = runtime(scenes::empty, false);
    let (x, y) = rt.find("size-5").unwrap();
    rt.pointer_down(x, y);
    rt.pointer_up(x, y);
    rt.frame(1.0 / 60.0);
    assert_eq!(rt.state.chosen_size(), Ok((3072, 5472)));
}

#[test]
fn the_custom_size_warns_until_it_is_a_multiple_of_32() {
    let mut rt = runtime(scenes::custom_size, false);
    assert!(text_of(&rt, "size-error").contains("multiples of 32"));
    rt.update(|s, _| {
        s.custom_w.replace_text("6016");
        s.custom_h.replace_text("2016");
    });
    for _ in 0..3 {
        rt.frame(1.0 / 60.0);
    }
    assert!(rt.find("size-error").is_none());
    assert_eq!(rt.state.chosen_size(), Ok((6016, 2016)));
}

#[test]
fn running_rows_show_their_step_and_the_preview_its_progress() {
    let rt = runtime(scenes::running, false);
    assert!(rt.find("progress-card").is_some());
    assert!(text_of(&rt, "progress-detail").ends_with("elapsed"));
}

#[test]
fn the_gallery_scene_shows_the_preview_stage_and_timings() {
    let rt = runtime(scenes::gallery, false);
    assert!(rt.find("artifact-stage").is_some());
    assert!(rt.find("timeline").is_some());
    assert!(text_of(&rt, "total").starts_with("Total"));
}

#[test]
fn gallery_thumbnails_are_decoded() {
    let rt = runtime(scenes::gallery, false);
    for g in &rt.state.gens {
        match &g.picture {
            state::Picture::Ready { thumb: Some(t), .. } => {
                assert!(t.is_ready() && t.error().is_none() && t.size().is_some(), "{}: {:?} {:?}", g.id, t.size(), t.error())
            }
            _ => panic!("{} has no thumbnail", g.id),
        }
    }
}

fn focused(rt: &Runtime<State>) -> Option<String> {
    rt.elements().into_iter().find(|e| e.focused).and_then(|e| e.key)
}

fn tap(rt: &mut Runtime<State>, key: &str) {
    let (x, y) = rt.find(key).unwrap_or_else(|| panic!("no {key}"));
    rt.pointer_down(x, y);
    rt.pointer_up(x, y);
    rt.frame(1.0 / 60.0);
}

#[test]
fn cmd_return_while_disconnected_adds_nothing() {
    let mut rt = runtime(scenes::disconnected, false);
    tap(&mut rt, "prompt");
    rt.update(|s, _| s.prompt.replace_text("a fox"));
    rt.frame(1.0 / 60.0);
    rt.key(KeyEvent { key: Key::Enter, mods: Mods { cmd: true, ..Mods::default() } });
    rt.frame(1.0 / 60.0);
    assert!(rt.state.gens.is_empty());
    assert_eq!(rt.state.notes.last().map(|n| n.title.as_str()), Some("Start mmh3 server first."));
}

#[test]
fn a_note_from_the_background_stays_up_after_an_idle_spell() {
    let mut rt = runtime(scenes::empty, false);
    rt.update(|s, _| s.note("Couldn't delete the generation.", None, true));
    rt.frame(60.0);
    assert_eq!(rt.state.notes.len(), 1, "a note isn't timed from a clock the window left behind");
    rt.frame(state::NOTE_SECONDS + 0.1);
    assert!(rt.state.notes.is_empty());
}

#[test]
fn the_size_tiles_are_one_tab_stop_and_arrows_move_between_them() {
    let mut rt = runtime(scenes::empty, false);
    tap(&mut rt, "size-0");
    assert_eq!(focused(&rt).as_deref(), Some("size-0"));
    rt.key(KeyEvent { key: Key::Right, mods: Mods::default() });
    rt.frame(1.0 / 60.0);
    assert_eq!(focused(&rt).as_deref(), Some("size-1"));
    rt.key(KeyEvent { key: Key::Tab, mods: Mods::default() });
    rt.frame(1.0 / 60.0);
    assert!(!focused(&rt).is_some_and(|k| k.starts_with("size-")), "Tab leaves the group: {:?}", focused(&rt));
}

#[test]
fn delete_asks_before_it_removes_anything() {
    let mut rt = runtime(scenes::gallery, false);
    let before = rt.state.gens.len();
    tap(&mut rt, "delete");
    for _ in 0..20 {
        rt.frame(1.0 / 60.0);
    }
    assert!(rt.find("delete-dialog").is_some());
    assert_eq!(rt.state.gens.len(), before);
    tap(&mut rt, "delete-cancel");
    assert_eq!(rt.state.deleting, None);
    assert_eq!(rt.state.gens.len(), before);
    tap(&mut rt, "delete");
    for _ in 0..20 {
        rt.frame(1.0 / 60.0);
    }
    tap(&mut rt, "delete-confirm");
    assert_eq!(rt.state.gens.len(), before - 1);
}

#[test]
fn the_status_bar_leaves_out_memory_the_server_could_not_measure() {
    let mut rt = runtime(scenes::empty, false);
    rt.update(|s, _| {
        if let state::Conn::Connected(st) = &mut s.conn {
            st.memory = None;
        }
    });
    rt.frame(1.0 / 60.0);
    assert!(rt.find("status-memory").is_none());
    assert!(text_of(&rt, "status-connection").starts_with("Connected"));
}

#[test]
fn the_offline_banner_offers_the_same_start_command_as_the_card() {
    let rt = runtime(scenes::offline_selected, false);
    assert!(rt.find("copy-start-command").is_some());
    assert!(!text_of(&rt, "offline-note").contains('`'));
}

#[test]
fn placeholder_cards_stay_inside_their_frame_at_the_smallest_window() {
    for scene in [scenes::failed as fn(&mut State), scenes::running] {
        let mut s = State::offline();
        scene(&mut s);
        let mut rt = Runtime::new(s, view::view, (1040.0, 680.0), Env { system_dark: false, scale: 1.0 });
        rt.poll = Some(poll);
        for _ in 0..30 {
            rt.frame(1.0 / 60.0);
        }
        let rect = |key: &str| rt.elements().into_iter().find(|e| e.key.as_deref() == Some(key)).map(|e| e.rect).unwrap();
        let (canvas, frame) = (rect("canvas"), rect("frame"));
        assert!(frame[0] >= canvas[0] && frame[0] + frame[2] <= canvas[0] + canvas[2], "frame {frame:?} spills out of {canvas:?}");
        let card = rt.elements().into_iter().find(|e| matches!(e.key.as_deref(), Some("failed" | "progress-card"))).unwrap().rect;
        assert!(card[0] >= frame[0] && card[0] + card[2] <= frame[0] + frame[2], "card {card:?} spills out of {frame:?}");
    }
}
